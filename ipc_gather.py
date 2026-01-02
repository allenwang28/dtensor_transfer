# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
DTensor IPC Transfer Demo

Demonstrates how to:
1. Create a tensor and shard it as DTensor A (Shard(0) - row-wise)
2. Transfer data via CUDA IPC handles
3. Reconstruct and reshard as DTensor B (Shard(1) - column-wise)

Uses Monarch actors to model the two different DTensor configurations.
"""

import logging
import os
import time
from dataclasses import dataclass
from functools import partial
from typing import Any, List, Tuple

import torch
from monarch.actor import Actor, current_rank, endpoint, this_host
from monarch.spmd import setup_torch_elastic_env
from torch.distributed._tensor import distribute_tensor, Shard
from torch.distributed.device_mesh import init_device_mesh

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_cuda_device(gpu_ids: List[int]) -> None:
    """Bootstrap function to set CUDA_VISIBLE_DEVICES based on rank.

    Each process gets exactly one GPU visible, so LOCAL_RANK must be 0
    for that process to use its single visible GPU correctly.
    """
    # Setup logging in the worker process
    logging.basicConfig(level=logging.INFO)

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    logger.info(f"[Rank {current_rank().rank}] Set CUDA_VISIBLE_DEVICES={gpu_ids}")


@dataclass
class CudaIPCHandle:
    """Serializable CUDA IPC handle for cross-process tensor sharing."""

    device: int
    handle: Any  # cudaIpcMemHandle_t (bytes)
    size_bytes: int
    offset_bytes: int
    ref_counter_handle: Any
    ref_counter_offset: int
    event_handle: Any
    event_sync_required: bool
    shape: Tuple[int, ...]
    stride: Tuple[int, ...]
    dtype: torch.dtype


class Sender(Actor):
    """Actor that creates a DTensor with Shard(0) and exposes IPC handles."""

    def __init__(self, global_tensor: torch.Tensor):
        self.rank = current_rank().rank
        self.global_tensor = global_tensor
        self.dtensor = None
        self.device_mesh = None

    @endpoint
    def setup_and_create_dtensor(self) -> Tuple[int, ...]:
        """Initialize distributed env, create DTensor with Shard(0)."""
        logger.info(f"[Sender {self.rank}] Initializing process group...")

        # Log distributed environment before init
        logger.info(f"[Sender {self.rank}] Env before init_process_group:")
        logger.info(f"[Sender {self.rank}]   RANK={os.environ.get('RANK')}")
        logger.info(f"[Sender {self.rank}]   LOCAL_RANK={os.environ.get('LOCAL_RANK')}")
        logger.info(f"[Sender {self.rank}]   WORLD_SIZE={os.environ.get('WORLD_SIZE')}")
        logger.info(
            f"[Sender {self.rank}]   MASTER_ADDR={os.environ.get('MASTER_ADDR')}"
        )
        logger.info(
            f"[Sender {self.rank}]   MASTER_PORT={os.environ.get('MASTER_PORT')}"
        )
        logger.info(
            f"[Sender {self.rank}]   CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
        )

        # Init process group - env vars set by setup_torch_elastic_env
        torch.distributed.init_process_group(backend="nccl")

        world_size = int(os.environ["WORLD_SIZE"])
        mesh_shape = (world_size,)

        logger.info(
            f"[Sender {self.rank}] Creating device mesh with shape {mesh_shape}"
        )
        self.device_mesh = init_device_mesh("cuda", mesh_shape)

        # Move tensor to CUDA and distribute with Shard(0)
        cuda_tensor = self.global_tensor.cuda()
        logger.info(
            f"[Sender {self.rank}] Distributing tensor with Shard(0), "
            f"global shape: {cuda_tensor.shape}"
        )

        self.dtensor = distribute_tensor(cuda_tensor, self.device_mesh, [Shard(0)])

        local_shape = tuple(self.dtensor.to_local().shape)
        logger.info(f"[Sender {self.rank}] Local shard shape: {local_shape}")

        return local_shape

    @endpoint
    def get_ipc_handle(self) -> CudaIPCHandle:
        """Extract CUDA IPC handle from local tensor shard."""
        local_tensor = self.dtensor.to_local()

        # Ensure tensor is contiguous for IPC
        if not local_tensor.is_contiguous():
            local_tensor = local_tensor.contiguous()

        storage = local_tensor.untyped_storage()

        # Get IPC handle from CUDA storage
        (
            device,
            handle,
            size_bytes,
            offset_bytes,
            ref_counter_handle,
            ref_counter_offset,
            event_handle,
            event_sync_required,
        ) = storage._share_cuda_()

        logger.info(f"[Sender {self.rank}] Extracted IPC handle for device {device}")

        return CudaIPCHandle(
            device=device,
            handle=handle,
            size_bytes=size_bytes,
            offset_bytes=offset_bytes,
            ref_counter_handle=ref_counter_handle,
            ref_counter_offset=ref_counter_offset,
            event_handle=event_handle,
            event_sync_required=event_sync_required,
            shape=tuple(local_tensor.shape),
            stride=tuple(local_tensor.stride()),
            dtype=local_tensor.dtype,
        )

    @endpoint
    def destroy(self) -> None:
        """Cleanup distributed resources."""
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        logger.info(f"[Sender {self.rank}] Process group destroyed")


class Receiver(Actor):
    """Actor that receives IPC handles and creates DTensor with Shard(1)."""

    def __init__(self, global_shape: Tuple[int, ...], dtype: torch.dtype):
        self.rank = current_rank().rank
        self.global_shape = global_shape
        self.dtype = dtype
        self.dtensor = None
        self.device_mesh = None

    @endpoint
    def init_process_group(self) -> None:
        """Initialize the process group for this receiver."""
        logger.info(f"[Receiver {self.rank}] Initializing process group...")
        torch.distributed.init_process_group(backend="nccl")

        world_size = int(os.environ["WORLD_SIZE"])
        mesh_shape = (world_size,)

        logger.info(
            f"[Receiver {self.rank}] Creating device mesh with shape {mesh_shape}"
        )
        self.device_mesh = init_device_mesh("cuda", mesh_shape)

    @endpoint
    def receive_and_reshard(self, ipc_handles: List[CudaIPCHandle]) -> Tuple[int, ...]:
        """
        Receive tensors via IPC handles, reconstruct full tensor,
        then redistribute with Shard(1).
        """
        # Reconstruct shards from IPC handles
        logger.info(
            f"[Receiver {self.rank}] Reconstructing {len(ipc_handles)} shards from IPC..."
        )

        # Get this receiver's local device
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        local_device = f"cuda:{local_rank}"

        shards = []

        for i, handle in enumerate(ipc_handles):
            logger.info(f"[Receiver {self.rank}]   Handle {i}: shape={handle.shape}, stride={handle.stride}, dtype={handle.dtype}, device={handle.device}")

            # Reconstruct storage from IPC handle
            storage = torch.UntypedStorage._new_shared_cuda(
                handle.device,
                handle.handle,
                handle.size_bytes,
                handle.offset_bytes,
                handle.ref_counter_handle,
                handle.ref_counter_offset,
                handle.event_handle,
                handle.event_sync_required,
            )

            # Create tensor from storage with correct shape/stride
            tensor = torch.empty(
                handle.shape, dtype=handle.dtype, device=f"cuda:{handle.device}"
            )
            # Use set_ with size and stride to properly reshape
            tensor.set_(storage, storage_offset=0, size=handle.shape, stride=handle.stride)

            # Copy to local device for concatenation
            tensor = tensor.to(local_device)

            logger.info(f"[Receiver {self.rank}]   Shard {i}: shape {tensor.shape} on {local_device}")
            shards.append(tensor)

        # Concatenate all shards to reconstruct full tensor
        # Original was Shard(0), so concat along dim 0
        full_tensor = torch.cat(shards, dim=0)
        logger.info(
            f"[Receiver {self.rank}] Reconstructed full tensor: {full_tensor.shape}"
        )

        # Create DTensor with new sharding: Shard(1)
        logger.info(f"[Receiver {self.rank}] Redistributing with Shard(1)...")
        self.dtensor = distribute_tensor(full_tensor, self.device_mesh, [Shard(1)])

        local_shape = tuple(self.dtensor.to_local().shape)
        logger.info(
            f"[Receiver {self.rank}] Local shard shape with Shard(1): {local_shape}"
        )

        return local_shape

    @endpoint
    def get_local_tensor(self) -> torch.Tensor:
        """Return the local tensor shard for verification."""
        return self.dtensor.to_local().cpu()

    @endpoint
    def destroy(self) -> None:
        """Cleanup distributed resources."""
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        logger.info(f"[Receiver {self.rank}] Process group destroyed")


def main():
    """
    Main function demonstrating DTensor transfer via CUDA IPC.

    1. Creates a tensor and shards it with Shard(0) across sender mesh
    2. Extracts IPC handles from each sender rank
    3. Passes handles to receiver mesh, which reconstructs and reshards with Shard(1)
    4. Benchmarks the transfer over multiple iterations
    """
    # Configuration
    n_senders = 2
    n_receivers = 2
    tensor_shape = (1024, 1024)  # Larger tensor for meaningful benchmarks
    n_warmup = 2
    n_iterations = 10

    # GPU assignment: senders get GPUs 0..n_senders-1, receivers get n_senders..n_senders+n_receivers-1
    sender_gpu_ids = list(range(n_senders))
    receiver_gpu_ids = list(range(n_senders, n_senders + n_receivers))

    logger.info("=" * 60)
    logger.info("DTensor CUDA IPC Transfer Demo")
    logger.info("=" * 60)
    logger.info(f"Senders: {n_senders} (GPUs {sender_gpu_ids})")
    logger.info(f"Receivers: {n_receivers} (GPUs {receiver_gpu_ids})")
    logger.info(f"Tensor shape: {tensor_shape}")
    logger.info(
        f"Tensor size: {tensor_shape[0] * tensor_shape[1] * 4 / 1024 / 1024:.2f} MB"
    )
    logger.info(f"Sender sharding: Shard(0) - row-wise")
    logger.info(f"Receiver sharding: Shard(1) - column-wise")
    logger.info(f"Warmup iterations: {n_warmup}")
    logger.info(f"Benchmark iterations: {n_iterations}")
    logger.info("=" * 60)

    # Create original tensor
    original = (
        torch.arange(tensor_shape[0] * tensor_shape[1]).reshape(tensor_shape).float()
    )
    logger.info(f"Created original tensor with shape {original.shape}")

    # Spawn proc meshes using this_host()
    logger.info("--- Spawning proc meshes ---")
    host = this_host()

    # Spawn sender procs with bootstrap to set CUDA_VISIBLE_DEVICES
    sender_procs = host.spawn_procs(
        per_host={"gpu": n_senders},
        bootstrap=partial(set_cuda_device, sender_gpu_ids),
    )
    logger.info("Sender procs created")

    # Setup distributed env for senders
    logger.info("Setting up distributed env for senders...")
    setup_torch_elastic_env(sender_procs)

    # Spawn sender actors
    senders = sender_procs.spawn("senders", Sender, original)

    # Create DTensor A with Shard(0)
    logger.info("--- Creating DTensor with Shard(0) ---")
    sender_shapes = senders.setup_and_create_dtensor.call().get()
    for proc_info, shape in sender_shapes:
        logger.info(f"  Sender rank {proc_info.rank}: local shape {shape}")

    # Get IPC handles from all senders
    logger.info("--- Extracting IPC handles ---")
    ipc_results = senders.get_ipc_handle.call().get()
    ipc_handles = [handle for _, handle in ipc_results]
    logger.info(f"  Collected {len(ipc_handles)} IPC handles")

    # Spawn receiver procs with bootstrap to set CUDA_VISIBLE_DEVICES
    receiver_procs = host.spawn_procs(
        per_host={"gpu": n_receivers},
        bootstrap=partial(set_cuda_device, receiver_gpu_ids),
    )
    logger.info("Receiver procs created")

    # Setup distributed env for receivers
    logger.info("Setting up distributed env for receivers...")
    setup_torch_elastic_env(receiver_procs)

    # Spawn receiver actors
    receivers = receiver_procs.spawn(
        "receivers", Receiver, tensor_shape, original.dtype
    )

    # Initialize receiver process groups once
    receivers.init_process_group.call().get()

    # Warmup iterations
    logger.info(f"--- Warmup ({n_warmup} iterations) ---")
    for i in range(n_warmup):
        start_time = time.perf_counter()
        receivers.receive_and_reshard.call(ipc_handles).get()
        elapsed = time.perf_counter() - start_time
        logger.info(f"  Warmup {i + 1}: {elapsed * 1000:.2f} ms")

    # Benchmark iterations
    logger.info(f"--- Benchmark ({n_iterations} iterations) ---")
    times = []
    for i in range(n_iterations):
        start_time = time.perf_counter()
        receivers.receive_and_reshard.call(ipc_handles).get()
        elapsed = time.perf_counter() - start_time
        times.append(elapsed)
        logger.info(f"  Iteration {i + 1}: {elapsed * 1000:.2f} ms")

    # Print benchmark results
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    tensor_size_mb = tensor_shape[0] * tensor_shape[1] * 4 / 1024 / 1024
    throughput = tensor_size_mb / avg_time

    logger.info("=" * 60)
    logger.info("BENCHMARK RESULTS")
    logger.info("=" * 60)
    logger.info(f"Tensor size:     {tensor_size_mb:.2f} MB")
    logger.info(f"Iterations:      {n_iterations}")
    logger.info(f"Avg time:        {avg_time * 1000:.2f} ms")
    logger.info(f"Min time:        {min_time * 1000:.2f} ms")
    logger.info(f"Max time:        {max_time * 1000:.2f} ms")
    logger.info(f"Throughput:      {throughput:.2f} MB/s")
    logger.info("=" * 60)

    # Verify correctness on final iteration
    logger.info("--- Verification ---")
    local_tensors = receivers.get_local_tensor.call().get()
    for proc_info, tensor in local_tensors:
        logger.info(
            f"  Receiver rank {proc_info.rank} local tensor shape: {tensor.shape}"
        )

    # Cleanup
    logger.info("--- Cleanup ---")
    try:
        senders.destroy.call().get()
    except Exception:
        pass

    try:
        receivers.destroy.call().get()
    except Exception:
        pass

    logger.info("SUCCESS: DTensor IPC transfer completed!")


if __name__ == "__main__":
    main()
