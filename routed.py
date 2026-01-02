# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
DTensor Direct Transfer Demo (Routed approach)

Unlike gather.py which gathers ALL shards then reshards, this approach:
1. Computes which sender shards overlap with which receiver shards
2. Senders pre-slice their tensors into exactly the chunks receivers need
3. RemoteTensor handles point to these pre-sliced chunks (not full shards)
4. Receivers copy chunks directly into the right positions

This is more efficient because:
- Only the exact data needed is transferred (no redundant data)
- Receivers don't need to gather entire sender shards
- Direct placement into destination tensor

Uses PyTorch DCP's resharding algorithm for overlap detection.
"""

import argparse
import logging
import os
from dataclasses import dataclass
from functools import partial
from typing import List, Tuple

# Enable expandable segments for better memory management
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
from monarch.actor import Actor, current_rank, endpoint, this_host
from monarch.spmd import setup_torch_elastic_env

from remote_tensor import RemoteTensor, Transport
from torch.distributed._tensor import DTensor, Shard
from torch.distributed.checkpoint.metadata import ChunkStorageMetadata
from torch.distributed.checkpoint.resharding import (
    _check_shard_metadata_pair_overlap,
    _shards_get_overlap_region_wrt_saved_tensor,
)
from torch.distributed.device_mesh import init_device_mesh

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class RankedChunk:
    """Wraps ChunkStorageMetadata with rank information."""

    rank: int  # Which rank owns this shard
    chunk: ChunkStorageMetadata  # DCP's chunk metadata (offsets, sizes)


@dataclass
class TransferChunk:
    """Describes a single chunk transfer from sender to receiver."""

    sender_rank: int
    receiver_rank: int
    sender_offset: Tuple[int, ...]  # Offset within sender's local shard
    receiver_offset: Tuple[int, ...]  # Offset within receiver's local shard
    lengths: Tuple[int, ...]  # Size of chunk to transfer


# =============================================================================
# Overlap Detection (using DCP utilities)
# =============================================================================


def compute_overlap(sender: RankedChunk, receiver: RankedChunk) -> TransferChunk | None:
    """
    Compute the overlap between a sender shard and receiver shard using DCP.

    Returns a TransferChunk if there's overlap, None otherwise.
    """
    if not _check_shard_metadata_pair_overlap(sender.chunk, receiver.chunk):
        return None

    overlap_info = _shards_get_overlap_region_wrt_saved_tensor(
        sender.chunk, receiver.chunk
    )

    sender_offsets = tuple(info[1] for info in overlap_info)
    receiver_offsets = tuple(info[2] for info in overlap_info)
    lengths = tuple(info[3] for info in overlap_info)

    return TransferChunk(
        sender_rank=sender.rank,
        receiver_rank=receiver.rank,
        sender_offset=sender_offsets,
        receiver_offset=receiver_offsets,
        lengths=lengths,
    )


def compute_transfer_plan(
    sender_shards: List[RankedChunk],
    receiver_shards: List[RankedChunk],
) -> dict[int, List[TransferChunk]]:
    """Compute which sender shards overlap with which receiver shards."""
    plan: dict[int, List[TransferChunk]] = {r.rank: [] for r in receiver_shards}

    for recv_shard in receiver_shards:
        for send_shard in sender_shards:
            chunk = compute_overlap(send_shard, recv_shard)
            if chunk is not None:
                plan[recv_shard.rank].append(chunk)

    return plan


def compute_sender_plan(
    sender_shards: List[RankedChunk],
    receiver_shards: List[RankedChunk],
) -> dict[int, List[TransferChunk]]:
    """Compute which chunks each sender needs to prepare."""
    plan: dict[int, List[TransferChunk]] = {s.rank: [] for s in sender_shards}

    for recv_shard in receiver_shards:
        for send_shard in sender_shards:
            chunk = compute_overlap(send_shard, recv_shard)
            if chunk is not None:
                plan[send_shard.rank].append(chunk)

    return plan


def compute_shard_metadata(
    global_shape: Tuple[int, ...],
    num_ranks: int,
    shard_dim: int,
) -> List[RankedChunk]:
    """Compute RankedChunk for each rank given a Shard(dim) placement."""
    shards = []
    dim_size = global_shape[shard_dim]
    chunk_size = (dim_size + num_ranks - 1) // num_ranks

    for rank in range(num_ranks):
        start = rank * chunk_size
        end = min(start + chunk_size, dim_size)
        actual_size = end - start

        offsets = [0] * len(global_shape)
        offsets[shard_dim] = start

        sizes = list(global_shape)
        sizes[shard_dim] = actual_size

        chunk_meta = ChunkStorageMetadata(
            offsets=torch.Size(offsets),
            sizes=torch.Size(sizes),
        )
        shards.append(RankedChunk(rank=rank, chunk=chunk_meta))

    return shards


# =============================================================================
# Tensor Creation
# =============================================================================


def create_local_shard(
    global_shape: Tuple[int, ...],
    rank: int,
    world_size: int,
    shard_dim: int,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda",
) -> torch.Tensor:
    """Create only the local shard for this rank."""
    num_cols = global_shape[1]
    dim_size = global_shape[shard_dim]
    chunk_size = (dim_size + world_size - 1) // world_size

    start = rank * chunk_size
    end = min(start + chunk_size, dim_size)

    if shard_dim == 0:
        row_indices = torch.arange(start, end, dtype=dtype, device=device).unsqueeze(1)
        col_indices = torch.arange(0, num_cols, dtype=dtype, device=device).unsqueeze(0)
        return row_indices * num_cols + col_indices
    else:
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
    """Create the expected values for a slice without allocating the full tensor."""
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
    """Bootstrap function to set CUDA_VISIBLE_DEVICES."""
    logging.basicConfig(level=logging.INFO)
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
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
# Actors
# =============================================================================


class Sender(Actor):
    """Actor that creates a DTensor with Shard(0) and exposes RemoteTensor handles."""

    def __init__(self, tensor_shape: Tuple[int, ...], dtype: torch.dtype):
        self.rank = current_rank().rank
        self.tensor_shape = tensor_shape
        self.dtype = dtype
        self.dtensor = None
        self.device_mesh = None
        # Store pre-sliced chunks to keep them alive
        self.chunk_tensors: dict[int, torch.Tensor] = {}

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

        logger.info(
            f"[Sender {self.rank}] Creating local shard for global shape {self.tensor_shape}"
        )
        local_shard = create_local_shard(
            self.tensor_shape,
            rank=dist_rank,
            world_size=world_size,
            shard_dim=0,
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
    def prepare_chunks(self, chunks: List[TransferChunk]) -> int:
        """Pre-slice the local tensor into chunks needed by receivers."""
        local_tensor = self.dtensor.to_local()
        self.chunk_tensors.clear()

        for chunk in chunks:
            slices = tuple(
                slice(off, off + length)
                for off, length in zip(chunk.sender_offset, chunk.lengths)
            )
            chunk_tensor = local_tensor[slices].contiguous()
            self.chunk_tensors[chunk.receiver_rank] = chunk_tensor

            logger.info(
                f"[Sender {self.rank}] Prepared chunk for receiver {chunk.receiver_rank}: "
                f"shape={chunk_tensor.shape}"
            )

        return len(self.chunk_tensors)

    @endpoint
    def get_chunk_remote_tensors(self) -> dict[int, RemoteTensor]:
        """Get RemoteTensor handles for all prepared chunks."""
        handles: dict[int, RemoteTensor] = {}

        physical_gpu = get_physical_gpu_id(self.rank)

        for receiver_rank, chunk_tensor in self.chunk_tensors.items():
            handle = RemoteTensor.from_tensor(
                chunk_tensor,
                owner=f"sender_{self.rank}_chunk_{receiver_rank}",
                enable_ipc=True,
                enable_rdma=True,
            )
            handles[receiver_rank] = handle

            logger.info(
                f"[Sender {self.rank}] Created RemoteTensor for receiver {receiver_rank}: "
                f"shape={chunk_tensor.shape}, GPU {physical_gpu}"
            )

        return handles

    @endpoint
    def destroy(self) -> None:
        """Cleanup distributed resources."""
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        logger.info(f"[Sender {self.rank}] Process group destroyed")


class Receiver(Actor):
    """Actor that receives specific chunks via direct transfer into a pre-allocated DTensor."""

    def __init__(self, global_shape: Tuple[int, ...], dtype: torch.dtype):
        self.rank = current_rank().rank
        self.global_shape = global_shape
        self.dtype = dtype
        self.dtensor = None
        self.device_mesh = None
        # List of (remote_tensor, receiver_slices)
        self.chunk_info: List[Tuple[RemoteTensor, Tuple[slice, ...]]] = []

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

        dim_size = self.global_shape[1]
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
    def setup_remote_tensors(
        self,
        chunk_handles: List[Tuple[int, RemoteTensor, Tuple[int, ...]]],
    ) -> int:
        """
        Store remote tensor handles for direct IPC access.

        Args:
            chunk_handles: List of (sender_rank, remote_tensor, receiver_offset) tuples.
        """
        self.chunk_info.clear()
        my_physical_gpu = get_physical_gpu_id(self.rank)

        for sender_rank, rt, receiver_offset in chunk_handles:
            # Build slice for receiver's local coordinates
            receiver_slices = tuple(
                slice(off, off + sz) for off, sz in zip(receiver_offset, rt.shape)
            )

            self.chunk_info.append((rt, receiver_slices))

            sender_physical_gpu = sender_rank
            logger.info(
                f"[Receiver {self.rank}] Set up RemoteTensor: "
                f"GPU {sender_physical_gpu} -> GPU {my_physical_gpu}, "
                f"shape={rt.shape}, offset={receiver_offset}"
            )

        return len(self.chunk_info)

    @endpoint
    def receive(self, n_streams: int = 1, transport: str = "auto") -> float:
        """
        Copy data from remote tensors into DTensor's local shard.
        Returns elapsed time in milliseconds.
        """
        transport_enum = Transport(transport)
        local_tensor = self.dtensor.to_local()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()

        if n_streams <= 1:
            for rt, receiver_slices in self.chunk_info:
                rt.read_into(local_tensor[receiver_slices], transport=transport_enum)
        else:
            streams = [torch.cuda.Stream() for _ in range(n_streams)]

            for i, (rt, receiver_slices) in enumerate(self.chunk_info):
                stream = streams[i % n_streams]
                with torch.cuda.stream(stream):
                    rt.read_into(
                        local_tensor[receiver_slices], transport=transport_enum
                    )

            for stream in streams:
                stream.synchronize()

        end_event.record()
        torch.cuda.synchronize()

        return start_event.elapsed_time(end_event)

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
    Main function demonstrating routed DTensor transfer via RemoteTensor.
    """
    parser = argparse.ArgumentParser(description="DTensor Transfer Demo (Routed)")
    parser.add_argument(
        "--transport",
        type=str,
        choices=["ipc", "rdma", "auto"],
        default="auto",
        help="Transport mechanism: ipc (same-node), rdma (cross-node), or auto",
    )
    args = parser.parse_args()

    # Configuration
    n_senders = 2
    n_receivers = 2
    tensor_shape = (51200, 51200)  # 10 GB tensor
    n_warmup = 2
    n_iterations = 10
    n_streams = 4
    transport = args.transport

    sender_shard_dim = 0
    receiver_shard_dim = 1

    sender_gpu_ids = list(range(n_senders))
    receiver_gpu_ids = list(range(n_senders, n_senders + n_receivers))

    logger.info("=" * 60)
    logger.info("DTensor Transfer Demo (Routed) - Using RemoteTensor")
    logger.info("=" * 60)
    logger.info(f"Senders: {n_senders} (GPUs {sender_gpu_ids})")
    logger.info(f"Receivers: {n_receivers} (GPUs {receiver_gpu_ids})")
    logger.info(f"Tensor shape: {tensor_shape}")
    logger.info(f"CUDA streams: {n_streams}")
    logger.info(f"Transport: {transport}")
    tensor_size_mb = tensor_shape[0] * tensor_shape[1] * 4 / 1024 / 1024
    logger.info(f"Tensor size: {tensor_size_mb:.2f} MB")
    logger.info(f"Sender sharding: Shard({sender_shard_dim}) - row-wise")
    logger.info(f"Receiver sharding: Shard({receiver_shard_dim}) - column-wise")
    logger.info(f"Warmup iterations: {n_warmup}")
    logger.info(f"Benchmark iterations: {n_iterations}")
    logger.info("=" * 60)

    # Compute shard metadata
    sender_shards = compute_shard_metadata(tensor_shape, n_senders, sender_shard_dim)
    receiver_shards = compute_shard_metadata(
        tensor_shape, n_receivers, receiver_shard_dim
    )

    logger.info("--- Sender Shard Layout ---")
    for s in sender_shards:
        logger.info(
            f"  Rank {s.rank}: offsets={tuple(s.chunk.offsets)}, sizes={tuple(s.chunk.sizes)}"
        )

    logger.info("--- Receiver Shard Layout ---")
    for r in receiver_shards:
        logger.info(
            f"  Rank {r.rank}: offsets={tuple(r.chunk.offsets)}, sizes={tuple(r.chunk.sizes)}"
        )

    # Compute transfer plans
    receiver_plan = compute_transfer_plan(sender_shards, receiver_shards)
    sender_plan = compute_sender_plan(sender_shards, receiver_shards)

    logger.info("--- Sender Plan (chunks to prepare) ---")
    for send_rank, chunks in sender_plan.items():
        logger.info(f"  Sender {send_rank} prepares {len(chunks)} chunks:")
        for chunk in chunks:
            logger.info(
                f"    For receiver {chunk.receiver_rank}: "
                f"offset={chunk.sender_offset}, len={chunk.lengths}"
            )

    logger.info("--- Receiver Plan (chunks to receive) ---")
    for recv_rank, chunks in receiver_plan.items():
        logger.info(f"  Receiver {recv_rank} needs {len(chunks)} chunks:")
        for chunk in chunks:
            logger.info(
                f"    From sender {chunk.sender_rank}: "
                f"recv_off={chunk.receiver_offset}, len={chunk.lengths}"
            )

    # Spawn proc meshes
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

    setup_torch_elastic_env(sender_procs)
    setup_torch_elastic_env(receiver_procs)

    dtype = torch.float32
    senders = sender_procs.spawn("senders", Sender, tensor_shape, dtype)
    receivers = receiver_procs.spawn("receivers", Receiver, tensor_shape, dtype)

    # Initialize DTensors in parallel
    logger.info("--- Creating DTensors ---")
    sender_init_future = senders.setup_and_create_dtensor.call()
    receiver_init_future = receivers.init_and_create_dtensor.call()

    sender_shapes = sender_init_future.get()
    for proc_info, shape in sender_shapes:
        logger.info(f"  Sender rank {proc_info.rank}: local shape {shape} (Shard(0))")

    receiver_shapes = receiver_init_future.get()
    for proc_info, shape in receiver_shapes:
        logger.info(f"  Receiver rank {proc_info.rank}: local shape {shape} (Shard(1))")

    # Have each sender prepare its chunks
    logger.info("--- Senders preparing chunks ---")
    for send_rank in range(n_senders):
        chunks = sender_plan[send_rank]
        senders.slice(gpu=send_rank).prepare_chunks.call(chunks).get()

    # Get RemoteTensor handles from all senders
    logger.info("--- Extracting RemoteTensor handles ---")
    chunk_handle_results = senders.get_chunk_remote_tensors.call().get()

    # Reorganize: receiver_rank -> list of (sender_rank, handle, receiver_offset)
    receiver_chunk_handles: dict[
        int, List[Tuple[int, RemoteTensor, Tuple[int, ...]]]
    ] = {r: [] for r in range(n_receivers)}

    for proc_info, handles_dict in chunk_handle_results:
        sender_rank = proc_info.rank
        for receiver_rank, handle in handles_dict.items():
            for chunk in sender_plan[sender_rank]:
                if chunk.receiver_rank == receiver_rank:
                    receiver_chunk_handles[receiver_rank].append(
                        (sender_rank, handle, chunk.receiver_offset)
                    )
                    break

    for recv_rank, handles in receiver_chunk_handles.items():
        logger.info(
            f"  Receiver {recv_rank} will receive {len(handles)} RemoteTensor handles"
        )

    # Set up remote tensors on receivers
    logger.info("--- Setting up RemoteTensors on receivers ---")
    for recv_rank in range(n_receivers):
        chunk_handles = receiver_chunk_handles[recv_rank]
        results = (
            receivers.slice(gpu=recv_rank)
            .setup_remote_tensors.call(chunk_handles)
            .get()
        )
        for proc_info, num_chunks in results:
            logger.info(
                f"  Receiver {proc_info.rank}: set up {num_chunks} remote tensors"
            )

    # Warmup
    logger.info(f"--- Warmup ({n_warmup} iterations) ---")
    for i in range(n_warmup):
        receivers.clear_tensor.call().get()
        results = receivers.receive.call(n_streams, transport).get()
        cuda_times = [elapsed_ms for _, elapsed_ms in results]
        max_cuda_time = max(cuda_times)
        logger.info(f"  Warmup {i + 1}: {max_cuda_time:.3f} ms (CUDA timed)")

    # Benchmark
    logger.info(f"--- Benchmark ({n_iterations} iterations) ---")
    times = []
    for i in range(n_iterations):
        receivers.clear_tensor.call().get()
        results = receivers.receive.call(n_streams, transport).get()
        cuda_times = [elapsed_ms for _, elapsed_ms in results]
        max_cuda_time = max(cuda_times)
        times.append(max_cuda_time)
        logger.info(f"  Iteration {i + 1}: {max_cuda_time:.3f} ms (CUDA timed)")

    # Results
    import statistics

    times_ms = times
    avg_time = statistics.mean(times_ms)
    min_time = min(times_ms)
    max_time = max(times_ms)
    median_time = statistics.median(times_ms)
    stdev_time = statistics.stdev(times_ms) if len(times_ms) > 1 else 0
    sorted_times = sorted(times_ms)
    p25 = sorted_times[len(sorted_times) // 4]
    p75 = sorted_times[(3 * len(sorted_times)) // 4]
    p90 = sorted_times[int(len(sorted_times) * 0.9)]
    throughput_mbs = tensor_size_mb / (avg_time / 1000)
    throughput_gbs = throughput_mbs / 1024

    logger.info("=" * 60)
    logger.info("BENCHMARK RESULTS (CUDA Event Timing)")
    logger.info("=" * 60)
    logger.info(f"Tensor size:     {tensor_size_mb:.2f} MB")
    logger.info(f"Transport:       {transport}")
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

    # Verify
    logger.info("--- Verification ---")
    local_tensors = receivers.get_local_tensor.call().get()
    for proc_info, tensor in local_tensors:
        logger.info(
            f"  Receiver rank {proc_info.rank} local tensor shape: {tensor.shape}"
        )
        recv_shard = receiver_shards[proc_info.rank]
        expected = create_expected_slice(
            tensor_shape,
            tuple(recv_shard.chunk.offsets),
            tuple(recv_shard.chunk.sizes),
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

    logger.info("SUCCESS: DTensor transfer completed!")


if __name__ == "__main__":
    main()
