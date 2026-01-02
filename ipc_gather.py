# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
DTensor IPC Transfer Demo (Gather approach)

Demonstrates how to:
1. Create a tensor and shard it as DTensor A (Shard(0) - row-wise)
2. Transfer data via CUDA IPC handles
3. Reconstruct and reshard as DTensor B (Shard(1) - column-wise)

This "gather" approach:
- Each receiver gathers ALL sender shards
- Concatenates them into the full tensor
- Then reshards with the new placement

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
from torch.distributed._tensor import DTensor, Shard
from torch.distributed.device_mesh import init_device_mesh

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Tensor Creation (called locally by senders, and for verification)
# =============================================================================


def create_local_shard(
    global_shape: Tuple[int, ...],
    rank: int,
    world_size: int,
    shard_dim: int,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Create only the local shard for this rank (avoids allocating full tensor).

    For tensor[i,j] = i * num_cols + j, computes the values for this rank's shard.
    """
    num_cols = global_shape[1]
    dim_size = global_shape[shard_dim]
    chunk_size = (dim_size + world_size - 1) // world_size

    start = rank * chunk_size
    end = min(start + chunk_size, dim_size)

    if shard_dim == 0:
        # Sharding rows: local shard is rows [start:end], all columns
        row_indices = torch.arange(start, end, dtype=dtype, device=device).unsqueeze(1)
        col_indices = torch.arange(0, num_cols, dtype=dtype, device=device).unsqueeze(0)
        return row_indices * num_cols + col_indices
    else:
        # Sharding columns: local shard is all rows, columns [start:end]
        num_rows = global_shape[0]
        row_indices = torch.arange(0, num_rows, dtype=dtype, device=device).unsqueeze(1)
        col_indices = torch.arange(start, end, dtype=dtype, device=device).unsqueeze(0)
        return row_indices * num_cols + col_indices


def create_expected_slice(
    global_shape: Tuple[int, ...],
    offsets: Tuple[int, ...],
    sizes: Tuple[int, ...],
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Create the expected values for a slice without allocating the full tensor.
    """
    num_cols = global_shape[1]
    row_off, col_off = offsets
    num_rows, num_cols_slice = sizes

    row_indices = torch.arange(row_off, row_off + num_rows, dtype=dtype).unsqueeze(1)
    col_indices = torch.arange(
        col_off, col_off + num_cols_slice, dtype=dtype
    ).unsqueeze(0)

    return row_indices * num_cols + col_indices


# =============================================================================
# Bootstrap Function
# =============================================================================


def set_cuda_device(gpu_ids: List[int]) -> None:
    """Bootstrap function to set CUDA_VISIBLE_DEVICES based on rank."""
    logging.basicConfig(level=logging.INFO)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    # Store physical GPU IDs so actors can reference them later
    os.environ["PHYSICAL_GPU_IDS"] = ",".join(str(g) for g in gpu_ids)
    rank = current_rank().rank
    physical_gpu = gpu_ids[rank] if rank < len(gpu_ids) else gpu_ids[0]
    logger.info(
        f"[Rank {rank}] Set CUDA_VISIBLE_DEVICES={gpu_ids}, using physical GPU {physical_gpu}"
    )


def get_physical_gpu_id(logical_device: int = 0) -> int:
    """Convert logical CUDA device index to physical GPU ID."""
    physical_ids = os.environ.get("PHYSICAL_GPU_IDS", "0").split(",")
    if logical_device < len(physical_ids):
        return int(physical_ids[logical_device])
    return logical_device


# =============================================================================
# Data Structures
# =============================================================================


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


# =============================================================================
# Actors
# =============================================================================


class Sender(Actor):
    """Actor that creates a DTensor with Shard(0) and exposes IPC handles."""

    def __init__(self, tensor_shape: Tuple[int, ...], dtype: torch.dtype):
        self.rank = current_rank().rank
        self.tensor_shape = tensor_shape
        self.dtype = dtype
        self.dtensor = None
        self.device_mesh = None

    @endpoint
    def setup_and_create_dtensor(self) -> Tuple[int, ...]:
        """Initialize distributed env, create DTensor with Shard(0)."""
        logger.info(f"[Sender {self.rank}] Initializing process group...")

        torch.distributed.init_process_group(backend="nccl")

        world_size = int(os.environ["WORLD_SIZE"])
        mesh_shape = (world_size,)
        dist_rank = int(os.environ["RANK"])

        logger.info(
            f"[Sender {self.rank}] Creating device mesh with shape {mesh_shape}"
        )
        self.device_mesh = init_device_mesh("cuda", mesh_shape)

        # Create only local shard directly on GPU (avoids allocating full tensor)
        logger.info(
            f"[Sender {self.rank}] Creating local shard for global shape {self.tensor_shape}"
        )
        local_shard = create_local_shard(
            self.tensor_shape,
            rank=dist_rank,
            world_size=world_size,
            shard_dim=0,  # Shard(0)
            dtype=self.dtype,
            device="cuda",
        )

        logger.info(
            f"[Sender {self.rank}] Wrapping as DTensor with Shard(0), "
            f"local shape: {local_shard.shape}"
        )

        self.dtensor = DTensor.from_local(
            local_shard, self.device_mesh, [Shard(0)], run_check=False
        )

        local_shape = tuple(self.dtensor.to_local().shape)
        logger.info(f"[Sender {self.rank}] Local shard shape: {local_shape}")

        return local_shape

    @endpoint
    def get_ipc_handle(self) -> CudaIPCHandle:
        """Extract CUDA IPC handle from local tensor shard."""
        local_tensor = self.dtensor.to_local()

        if not local_tensor.is_contiguous():
            local_tensor = local_tensor.contiguous()

        storage = local_tensor.untyped_storage()

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

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        physical_gpu = get_physical_gpu_id(local_rank)
        logger.info(
            f"[Sender {self.rank}] Extracted IPC handle for physical GPU {physical_gpu}"
        )

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
        # Pre-opened IPC tensors for repeated transfers
        self.ipc_shards: List[torch.Tensor] = []

    @endpoint
    def init_and_create_dtensor(self) -> Tuple[int, ...]:
        """Initialize process group and create DTensor with Shard(1)."""
        logger.info(f"[Receiver {self.rank}] Initializing process group...")
        torch.distributed.init_process_group(backend="nccl")

        world_size = int(os.environ["WORLD_SIZE"])
        mesh_shape = (world_size,)
        dist_rank = int(os.environ["RANK"])

        logger.info(
            f"[Receiver {self.rank}] Creating device mesh with shape {mesh_shape}"
        )
        self.device_mesh = init_device_mesh("cuda", mesh_shape)

        # Compute local shard size for Shard(1) and allocate only that
        dim_size = self.global_shape[1]  # Sharding columns
        chunk_size = (dim_size + world_size - 1) // world_size
        start = dist_rank * chunk_size
        end = min(start + chunk_size, dim_size)
        local_cols = end - start
        local_shape = (self.global_shape[0], local_cols)

        logger.info(
            f"[Receiver {self.rank}] Creating local shard with shape {local_shape} "
            f"(global: {self.global_shape})"
        )
        local_shard = torch.zeros(local_shape, dtype=self.dtype, device="cuda")

        self.dtensor = DTensor.from_local(
            local_shard, self.device_mesh, [Shard(1)], run_check=False
        )

        logger.info(f"[Receiver {self.rank}] Local shard shape: {local_shape}")

        return local_shape

    @endpoint
    def setup_ipc_handles(self, ipc_handles: List[CudaIPCHandle]) -> int:
        """
        Open IPC handles once and store references for repeated transfers.
        Returns the number of shards set up.
        """
        self.ipc_shards.clear()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        my_physical_gpu = get_physical_gpu_id(local_rank)

        for i, handle in enumerate(ipc_handles):
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

            # Create tensor from storage
            tensor = torch.empty(
                handle.shape, dtype=handle.dtype, device=f"cuda:{handle.device}"
            )
            tensor.set_(
                storage, storage_offset=0, size=handle.shape, stride=handle.stride
            )

            self.ipc_shards.append(tensor)

            # Sender i maps to physical GPU i (for senders using GPUs 0..n_senders-1)
            sender_physical_gpu = i
            logger.info(
                f"[Receiver {self.rank}] Set up IPC shard {i}: "
                f"GPU {sender_physical_gpu} -> GPU {my_physical_gpu}, "
                f"shape={handle.shape}"
            )

        return len(self.ipc_shards)

    @endpoint
    def receive(self, n_streams: int = 1) -> float:
        """
        Gather all shards, concatenate, and reshard into DTensor.
        Returns elapsed time in milliseconds (CUDA event timing).
        """
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        local_device = f"cuda:{local_rank}"

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()

        if n_streams <= 1:
            # Sequential copy
            shards = []
            for tensor in self.ipc_shards:
                shards.append(tensor.to(local_device))
        else:
            # Parallel copy with multiple streams
            streams = [torch.cuda.Stream() for _ in range(n_streams)]
            shards = [None] * len(self.ipc_shards)

            for i, tensor in enumerate(self.ipc_shards):
                stream = streams[i % n_streams]
                with torch.cuda.stream(stream):
                    shards[i] = tensor.to(local_device)

            # Sync all streams
            for stream in streams:
                stream.synchronize()

        # Concatenate all shards (Shard(0) means concat along dim 0)
        full_tensor = torch.cat(shards, dim=0)

        # Reshard into our local DTensor with Shard(1)
        # This requires slicing out our column portion
        world_size = int(os.environ["WORLD_SIZE"])
        dist_rank = int(os.environ["RANK"])
        dim_size = self.global_shape[1]
        chunk_size = (dim_size + world_size - 1) // world_size
        col_start = dist_rank * chunk_size
        col_end = min(col_start + chunk_size, dim_size)

        # Copy our column slice into the DTensor's local storage
        self.dtensor.to_local().copy_(full_tensor[:, col_start:col_end])

        end_event.record()
        torch.cuda.synchronize()

        return start_event.elapsed_time(end_event)  # ms

    @endpoint
    def clear_tensor(self) -> None:
        """Zero out the local tensor for the next benchmark run."""
        self.dtensor.to_local().zero_()

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


# =============================================================================
# Main
# =============================================================================


def main():
    """
    Main function demonstrating DTensor transfer via CUDA IPC (gather approach).

    1. Creates a tensor and shards it with Shard(0) across sender mesh
    2. Extracts IPC handles from each sender rank
    3. Passes handles to receiver mesh, which gathers, concatenates, reshards with Shard(1)
    4. Benchmarks the transfer over multiple iterations
    """
    # Configuration
    n_senders = 2
    n_receivers = 2
    tensor_shape = (51200, 51200)  # 10 GB tensor
    n_warmup = 2
    n_iterations = 10
    n_streams = 4  # Number of CUDA streams for parallel copies

    sender_gpu_ids = list(range(n_senders))
    receiver_gpu_ids = list(range(n_senders, n_senders + n_receivers))

    logger.info("=" * 60)
    logger.info("DTensor CUDA IPC Transfer Demo (Gather)")
    logger.info("=" * 60)
    logger.info(f"Senders: {n_senders} (GPUs {sender_gpu_ids})")
    logger.info(f"Receivers: {n_receivers} (GPUs {receiver_gpu_ids})")
    logger.info(f"Tensor shape: {tensor_shape}")
    logger.info(f"CUDA streams: {n_streams}")
    tensor_size_mb = tensor_shape[0] * tensor_shape[1] * 4 / 1024 / 1024
    logger.info(f"Tensor size: {tensor_size_mb:.2f} MB")
    logger.info("Sender sharding: Shard(0) - row-wise")
    logger.info("Receiver sharding: Shard(1) - column-wise")
    logger.info(f"Warmup iterations: {n_warmup}")
    logger.info(f"Benchmark iterations: {n_iterations}")
    logger.info("=" * 60)

    # Spawn both proc meshes in parallel
    logger.info("--- Spawning proc meshes ---")
    host = this_host()

    sender_procs = host.spawn_procs(
        per_host={"gpu": n_senders},
        bootstrap=partial(set_cuda_device, sender_gpu_ids),
    )
    receiver_procs = host.spawn_procs(
        per_host={"gpu": n_receivers},
        bootstrap=partial(set_cuda_device, receiver_gpu_ids),
    )
    logger.info("Sender and receiver procs created")

    # Setup distributed env for both in parallel
    setup_torch_elastic_env(sender_procs)
    setup_torch_elastic_env(receiver_procs)

    # Spawn actors (pass shape/dtype, tensors created locally on GPUs)
    dtype = torch.float32
    senders = sender_procs.spawn("senders", Sender, tensor_shape, dtype)
    receivers = receiver_procs.spawn("receivers", Receiver, tensor_shape, dtype)

    # Initialize both sender and receiver DTensors in parallel
    logger.info("--- Creating DTensors ---")
    sender_init_future = senders.setup_and_create_dtensor.call()
    receiver_init_future = receivers.init_and_create_dtensor.call()

    sender_shapes = sender_init_future.get()
    for proc_info, shape in sender_shapes:
        logger.info(f"  Sender rank {proc_info.rank}: local shape {shape} (Shard(0))")

    receiver_shapes = receiver_init_future.get()
    for proc_info, shape in receiver_shapes:
        logger.info(f"  Receiver rank {proc_info.rank}: local shape {shape} (Shard(1))")

    # Get IPC handles from all senders
    logger.info("--- Extracting IPC handles ---")
    ipc_results = senders.get_ipc_handle.call().get()
    ipc_handles = [handle for _, handle in ipc_results]
    logger.info(f"  Collected {len(ipc_handles)} IPC handles")

    # Set up IPC handles once on all receivers
    logger.info("--- Setting up IPC handles on receivers ---")
    setup_results = receivers.setup_ipc_handles.call(ipc_handles).get()
    for proc_info, num_shards in setup_results:
        logger.info(f"  Receiver {proc_info.rank}: set up {num_shards} IPC shards")

    # Warmup iterations
    logger.info(f"--- Warmup ({n_warmup} iterations) ---")
    for i in range(n_warmup):
        receivers.clear_tensor.call().get()
        results = receivers.receive.call(n_streams).get()
        cuda_times = [elapsed_ms for _, elapsed_ms in results]
        max_cuda_time = max(cuda_times)
        logger.info(f"  Warmup {i + 1}: {max_cuda_time:.3f} ms (CUDA timed)")

    # Benchmark iterations
    logger.info(f"--- Benchmark ({n_iterations} iterations) ---")
    times = []
    for i in range(n_iterations):
        receivers.clear_tensor.call().get()
        results = receivers.receive.call(n_streams).get()
        cuda_times = [elapsed_ms for _, elapsed_ms in results]
        max_cuda_time = max(cuda_times)
        times.append(max_cuda_time)
        logger.info(f"  Iteration {i + 1}: {max_cuda_time:.3f} ms (CUDA timed)")

    # Print benchmark results
    import statistics

    times_ms = times  # Already in ms from CUDA events
    avg_time = statistics.mean(times_ms)
    min_time = min(times_ms)
    max_time = max(times_ms)
    median_time = statistics.median(times_ms)
    stdev_time = statistics.stdev(times_ms) if len(times_ms) > 1 else 0
    sorted_times = sorted(times_ms)
    p25 = sorted_times[len(sorted_times) // 4]
    p75 = sorted_times[(3 * len(sorted_times)) // 4]
    p90 = sorted_times[int(len(sorted_times) * 0.9)]
    throughput_mbs = tensor_size_mb / (avg_time / 1000)  # MB/s
    throughput_gbs = throughput_mbs / 1024  # GB/s

    logger.info("=" * 60)
    logger.info("BENCHMARK RESULTS (CUDA Event Timing)")
    logger.info("=" * 60)
    logger.info(f"Tensor size:     {tensor_size_mb:.2f} MB")
    logger.info(f"Iterations:      {n_iterations}")
    logger.info("")
    logger.info("All times (ms):")
    for i, t in enumerate(times_ms):
        logger.info(f"  [{i+1:2d}] {t:8.3f} ms")
    logger.info("")
    logger.info("Distribution (ms):")
    logger.info(f"  Min:           {min_time:.3f}")
    logger.info(f"  P25:           {p25:.3f}")
    logger.info(f"  Median (P50):  {median_time:.3f}")
    logger.info(f"  P75:           {p75:.3f}")
    logger.info(f"  P90:           {p90:.3f}")
    logger.info(f"  Max:           {max_time:.3f}")
    logger.info("")
    logger.info(f"  Mean:          {avg_time:.3f}")
    logger.info(f"  Std Dev:       {stdev_time:.3f}")
    logger.info("")
    logger.info(
        f"Throughput:      {throughput_mbs:.2f} MB/s ({throughput_gbs:.2f} GB/s)"
    )
    logger.info("=" * 60)

    # Verify correctness
    logger.info("--- Verification ---")
    local_tensors = receivers.get_local_tensor.call().get()

    # Compute expected values for each receiver's shard
    world_size = n_receivers
    for proc_info, tensor in local_tensors:
        logger.info(
            f"  Receiver rank {proc_info.rank} local tensor shape: {tensor.shape}"
        )
        # Compute expected slice for this receiver
        dim_size = tensor_shape[1]
        chunk_size = (dim_size + world_size - 1) // world_size
        col_start = proc_info.rank * chunk_size
        col_end = min(col_start + chunk_size, dim_size)

        expected = create_expected_slice(
            tensor_shape,
            (0, col_start),
            (tensor_shape[0], col_end - col_start),
            dtype,
        )
        if torch.equal(tensor, expected):
            logger.info(f"  Receiver rank {proc_info.rank}: VERIFIED ✓")
        else:
            logger.error(f"  Receiver rank {proc_info.rank}: MISMATCH ✗")

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
