# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
DTensor IPC Routed Transfer Demo

Unlike ipc_gather.py which gathers ALL shards then reshards, this approach:
1. Computes which sender shards overlap with which receiver shards
2. Each receiver only reads the specific chunks it needs from sender shards
3. Directly assembles the local shard without full tensor reconstruction

This is more efficient for large tensors as it avoids redundant data transfer.

Inspired by PyTorch DCP's resharding algorithm using overlap detection.
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


@dataclass
class ShardMetadata:
    """Describes a shard's position in the global tensor."""

    rank: int  # Which rank owns this shard
    offsets: Tuple[int, ...]  # Global offset (e.g., (512, 0) for row 512)
    sizes: Tuple[int, ...]  # Size of this shard (e.g., (512, 1024))


@dataclass
class TransferChunk:
    """Describes a single chunk transfer from sender to receiver."""

    sender_rank: int
    sender_offset: Tuple[int, ...]  # Offset within sender's local shard
    receiver_offset: Tuple[int, ...]  # Offset within receiver's local shard
    lengths: Tuple[int, ...]  # Size of chunk to transfer


# =============================================================================
# Overlap Detection Algorithm (inspired by DCP)
# =============================================================================


def compute_overlap(
    sender: ShardMetadata, receiver: ShardMetadata
) -> TransferChunk | None:
    """
    Compute the overlap between a sender shard and receiver shard.

    Returns a TransferChunk if there's overlap, None otherwise.
    """
    ndim = len(sender.offsets)
    assert ndim == len(receiver.offsets) == len(sender.sizes) == len(receiver.sizes)

    # Compute overlap in each dimension
    sender_offsets = []
    receiver_offsets = []
    lengths = []

    for dim in range(ndim):
        # Sender's range in global coordinates
        s_start = sender.offsets[dim]
        s_end = s_start + sender.sizes[dim]

        # Receiver's range in global coordinates
        r_start = receiver.offsets[dim]
        r_end = r_start + receiver.sizes[dim]

        # Compute overlap
        overlap_start = max(s_start, r_start)
        overlap_end = min(s_end, r_end)

        if overlap_start >= overlap_end:
            # No overlap in this dimension
            return None

        # Offsets relative to each shard's local coordinates
        sender_offsets.append(overlap_start - s_start)
        receiver_offsets.append(overlap_start - r_start)
        lengths.append(overlap_end - overlap_start)

    return TransferChunk(
        sender_rank=sender.rank,
        sender_offset=tuple(sender_offsets),
        receiver_offset=tuple(receiver_offsets),
        lengths=tuple(lengths),
    )


def compute_transfer_plan(
    sender_shards: List[ShardMetadata],
    receiver_shards: List[ShardMetadata],
) -> dict[int, List[TransferChunk]]:
    """
    Compute which sender shards overlap with which receiver shards.

    Returns a dict mapping receiver_rank -> list of TransferChunks.
    """
    plan: dict[int, List[TransferChunk]] = {r.rank: [] for r in receiver_shards}

    for recv_shard in receiver_shards:
        for send_shard in sender_shards:
            chunk = compute_overlap(send_shard, recv_shard)
            if chunk is not None:
                plan[recv_shard.rank].append(chunk)

    return plan


def compute_shard_metadata(
    global_shape: Tuple[int, ...],
    num_ranks: int,
    shard_dim: int,
) -> List[ShardMetadata]:
    """
    Compute ShardMetadata for each rank given a Shard(dim) placement.
    """
    shards = []
    dim_size = global_shape[shard_dim]
    chunk_size = (dim_size + num_ranks - 1) // num_ranks  # Ceiling division

    for rank in range(num_ranks):
        start = rank * chunk_size
        end = min(start + chunk_size, dim_size)
        actual_size = end - start

        offsets = [0] * len(global_shape)
        offsets[shard_dim] = start

        sizes = list(global_shape)
        sizes[shard_dim] = actual_size

        shards.append(
            ShardMetadata(rank=rank, offsets=tuple(offsets), sizes=tuple(sizes))
        )

    return shards


# =============================================================================
# Bootstrap Function
# =============================================================================


def set_cuda_device(gpu_ids: List[int]) -> None:
    """Bootstrap function to set CUDA_VISIBLE_DEVICES."""
    logging.basicConfig(level=logging.INFO)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    logger.info(f"[Rank {current_rank().rank}] Set CUDA_VISIBLE_DEVICES={gpu_ids}")


# =============================================================================
# Actors
# =============================================================================


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

        torch.distributed.init_process_group(backend="nccl")

        world_size = int(os.environ["WORLD_SIZE"])
        mesh_shape = (world_size,)

        logger.info(
            f"[Sender {self.rank}] Creating device mesh with shape {mesh_shape}"
        )
        self.device_mesh = init_device_mesh("cuda", mesh_shape)

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
    """Actor that receives specific chunks via routed transfer."""

    def __init__(self, global_shape: Tuple[int, ...], dtype: torch.dtype):
        self.rank = current_rank().rank
        self.global_shape = global_shape
        self.dtype = dtype
        self.local_tensor = None
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
    def receive_routed(
        self,
        ipc_handles: List[CudaIPCHandle],
        transfer_chunks: List[TransferChunk],
        local_shape: Tuple[int, ...],
    ) -> Tuple[int, ...]:
        """
        Receive only the specific chunks needed for this receiver's shard.

        Instead of gathering all data and resharding, we:
        1. Allocate our local shard tensor
        2. For each TransferChunk, copy only the needed region from the sender
        """
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        local_device = f"cuda:{local_rank}"

        logger.info(
            f"[Receiver {self.rank}] Receiving {len(transfer_chunks)} chunks "
            f"into local tensor of shape {local_shape}"
        )

        # Allocate local tensor for this receiver's shard
        self.local_tensor = torch.zeros(
            local_shape, dtype=self.dtype, device=local_device
        )

        # Process each transfer chunk
        for chunk in transfer_chunks:
            logger.info(
                f"[Receiver {self.rank}]   Chunk from sender {chunk.sender_rank}: "
                f"sender_offset={chunk.sender_offset}, "
                f"receiver_offset={chunk.receiver_offset}, "
                f"lengths={chunk.lengths}"
            )

            handle = ipc_handles[chunk.sender_rank]

            # Reconstruct sender's tensor from IPC handle
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

            sender_tensor = torch.empty(
                handle.shape, dtype=handle.dtype, device=f"cuda:{handle.device}"
            )
            sender_tensor.set_(
                storage, storage_offset=0, size=handle.shape, stride=handle.stride
            )

            # Extract the specific chunk from sender
            # Build slice for sender's local coordinates
            sender_slices = tuple(
                slice(off, off + length)
                for off, length in zip(chunk.sender_offset, chunk.lengths)
            )
            sender_chunk = sender_tensor[sender_slices]

            # Copy to local device if needed
            sender_chunk = sender_chunk.to(local_device)

            # Build slice for receiver's local coordinates
            receiver_slices = tuple(
                slice(off, off + length)
                for off, length in zip(chunk.receiver_offset, chunk.lengths)
            )

            # Copy chunk to the correct position in local tensor
            self.local_tensor[receiver_slices] = sender_chunk

        logger.info(f"[Receiver {self.rank}] Routed transfer complete")
        return tuple(self.local_tensor.shape)

    @endpoint
    def get_local_tensor(self) -> torch.Tensor:
        """Return the local tensor shard for verification."""
        return self.local_tensor.cpu()

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
    Main function demonstrating routed DTensor transfer via CUDA IPC.

    Key difference from ipc_gather.py:
    - Computes overlap between Shard(0) and Shard(1) layouts
    - Each receiver only reads the chunks it needs (no full gather)
    """
    # Configuration
    n_senders = 2
    n_receivers = 2
    tensor_shape = (1024, 1024)
    n_warmup = 2
    n_iterations = 10

    sender_shard_dim = 0  # Shard(0) - row-wise
    receiver_shard_dim = 1  # Shard(1) - column-wise

    sender_gpu_ids = list(range(n_senders))
    receiver_gpu_ids = list(range(n_senders, n_senders + n_receivers))

    logger.info("=" * 60)
    logger.info("DTensor CUDA IPC Routed Transfer Demo")
    logger.info("=" * 60)
    logger.info(f"Senders: {n_senders} (GPUs {sender_gpu_ids})")
    logger.info(f"Receivers: {n_receivers} (GPUs {receiver_gpu_ids})")
    logger.info(f"Tensor shape: {tensor_shape}")
    logger.info(
        f"Tensor size: {tensor_shape[0] * tensor_shape[1] * 4 / 1024 / 1024:.2f} MB"
    )
    logger.info(f"Sender sharding: Shard({sender_shard_dim}) - row-wise")
    logger.info(f"Receiver sharding: Shard({receiver_shard_dim}) - column-wise")
    logger.info(f"Warmup iterations: {n_warmup}")
    logger.info(f"Benchmark iterations: {n_iterations}")
    logger.info("=" * 60)

    # Compute shard metadata for both layouts
    sender_shards = compute_shard_metadata(tensor_shape, n_senders, sender_shard_dim)
    receiver_shards = compute_shard_metadata(
        tensor_shape, n_receivers, receiver_shard_dim
    )

    logger.info("--- Sender Shard Layout ---")
    for s in sender_shards:
        logger.info(f"  Rank {s.rank}: offsets={s.offsets}, sizes={s.sizes}")

    logger.info("--- Receiver Shard Layout ---")
    for r in receiver_shards:
        logger.info(f"  Rank {r.rank}: offsets={r.offsets}, sizes={r.sizes}")

    # Compute transfer plan
    transfer_plan = compute_transfer_plan(sender_shards, receiver_shards)

    logger.info("--- Transfer Plan ---")
    for recv_rank, chunks in transfer_plan.items():
        logger.info(f"  Receiver {recv_rank} needs {len(chunks)} chunks:")
        for chunk in chunks:
            logger.info(
                f"    From sender {chunk.sender_rank}: "
                f"sender_off={chunk.sender_offset}, "
                f"recv_off={chunk.receiver_offset}, "
                f"len={chunk.lengths}"
            )

    # Create original tensor
    original = (
        torch.arange(tensor_shape[0] * tensor_shape[1]).reshape(tensor_shape).float()
    )
    logger.info(f"Created original tensor with shape {original.shape}")

    # Spawn proc meshes
    logger.info("--- Spawning proc meshes ---")
    host = this_host()

    sender_procs = host.spawn_procs(
        per_host={"gpu": n_senders},
        bootstrap=partial(set_cuda_device, sender_gpu_ids),
    )
    logger.info("Sender procs created")

    setup_torch_elastic_env(sender_procs)
    senders = sender_procs.spawn("senders", Sender, original)

    logger.info("--- Creating DTensor with Shard(0) ---")
    sender_shapes = senders.setup_and_create_dtensor.call().get()
    for proc_info, shape in sender_shapes:
        logger.info(f"  Sender rank {proc_info.rank}: local shape {shape}")

    logger.info("--- Extracting IPC handles ---")
    ipc_results = senders.get_ipc_handle.call().get()
    ipc_handles = [handle for _, handle in ipc_results]
    logger.info(f"  Collected {len(ipc_handles)} IPC handles")

    # Spawn receivers
    receiver_procs = host.spawn_procs(
        per_host={"gpu": n_receivers},
        bootstrap=partial(set_cuda_device, receiver_gpu_ids),
    )
    logger.info("Receiver procs created")

    setup_torch_elastic_env(receiver_procs)
    receivers = receiver_procs.spawn(
        "receivers", Receiver, tensor_shape, original.dtype
    )

    receivers.init_process_group.call().get()

    # Warmup iterations
    logger.info(f"--- Warmup ({n_warmup} iterations) ---")
    for i in range(n_warmup):
        start_time = time.perf_counter()

        # Each receiver gets its specific chunks
        # For simplicity, we call each receiver with its plan
        # In practice, you'd want to batch this
        for recv_rank in range(n_receivers):
            recv_chunks = transfer_plan[recv_rank]
            local_shape = receiver_shards[recv_rank].sizes
            receivers.slice(gpu=recv_rank).receive_routed.call(
                ipc_handles, recv_chunks, local_shape
            ).get()

        elapsed = time.perf_counter() - start_time
        logger.info(f"  Warmup {i + 1}: {elapsed * 1000:.2f} ms")

    # Benchmark iterations
    logger.info(f"--- Benchmark ({n_iterations} iterations) ---")
    times = []
    for i in range(n_iterations):
        start_time = time.perf_counter()

        for recv_rank in range(n_receivers):
            recv_chunks = transfer_plan[recv_rank]
            local_shape = receiver_shards[recv_rank].sizes
            receivers.slice(gpu=recv_rank).receive_routed.call(
                ipc_handles, recv_chunks, local_shape
            ).get()

        elapsed = time.perf_counter() - start_time
        times.append(elapsed)
        logger.info(f"  Iteration {i + 1}: {elapsed * 1000:.2f} ms")

    # Print benchmark results
    import statistics

    times_ms = [t * 1000 for t in times]
    avg_time = statistics.mean(times_ms)
    min_time = min(times_ms)
    max_time = max(times_ms)
    median_time = statistics.median(times_ms)
    stdev_time = statistics.stdev(times_ms) if len(times_ms) > 1 else 0
    sorted_times = sorted(times_ms)
    p25 = sorted_times[len(sorted_times) // 4]
    p75 = sorted_times[(3 * len(sorted_times)) // 4]
    p90 = sorted_times[int(len(sorted_times) * 0.9)]
    tensor_size_mb = tensor_shape[0] * tensor_shape[1] * 4 / 1024 / 1024
    throughput = tensor_size_mb / (avg_time / 1000)

    logger.info("=" * 60)
    logger.info("BENCHMARK RESULTS")
    logger.info("=" * 60)
    logger.info(f"Tensor size:     {tensor_size_mb:.2f} MB")
    logger.info(f"Iterations:      {n_iterations}")
    logger.info("")
    logger.info("All times (ms):")
    for i, t in enumerate(times_ms):
        logger.info(f"  [{i+1:2d}] {t:8.2f} ms")
    logger.info("")
    logger.info("Distribution (ms):")
    logger.info(f"  Min:           {min_time:.2f}")
    logger.info(f"  P25:           {p25:.2f}")
    logger.info(f"  Median (P50):  {median_time:.2f}")
    logger.info(f"  P75:           {p75:.2f}")
    logger.info(f"  P90:           {p90:.2f}")
    logger.info(f"  Max:           {max_time:.2f}")
    logger.info("")
    logger.info(f"  Mean:          {avg_time:.2f}")
    logger.info(f"  Std Dev:       {stdev_time:.2f}")
    logger.info("")
    logger.info(f"Throughput:      {throughput:.2f} MB/s")
    logger.info("=" * 60)

    # Verify correctness
    logger.info("--- Verification ---")
    local_tensors = receivers.get_local_tensor.call().get()
    for proc_info, tensor in local_tensors:
        logger.info(
            f"  Receiver rank {proc_info.rank} local tensor shape: {tensor.shape}"
        )
        # Verify content matches expected shard of original
        recv_shard = receiver_shards[proc_info.rank]
        expected_slices = tuple(
            slice(off, off + sz)
            for off, sz in zip(recv_shard.offsets, recv_shard.sizes)
        )
        expected = original[expected_slices]
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

    logger.info("SUCCESS: DTensor IPC routed transfer completed!")


if __name__ == "__main__":
    main()
