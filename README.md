# DTensor Transfer

Exploring how to efficiently transfer data between DTensors with different sharding layouts across GPUs.

## Overview

When working with distributed tensors (DTensors), a common operation is **resharding** - changing how a tensor is partitioned across devices. For example, transitioning from row-wise sharding (`Shard(0)`) to column-wise sharding (`Shard(1)`) between different model parallelism configurations.

## Approaches

### Gather (`ipc_gather.py`)

Each receiver gathers **all** sender shards, concatenates them into the full tensor, then extracts its portion:

```
Sender 0 (rows 0-511)    ──┐
                           ├──▶ Receiver 0: gather all → concat → slice cols 0-511
Sender 1 (rows 512-1023) ──┘

Sender 0 (rows 0-511)    ──┐
                           ├──▶ Receiver 1: gather all → concat → slice cols 512-1023
Sender 1 (rows 512-1023) ──┘
```

**Pros**: Simple implementation
**Cons**: Each receiver transfers the entire tensor, even though it only needs half

### Routed / Direct (`ipc_routed.py`)

Senders pre-slice their tensors into exactly the chunks each receiver needs. Only the necessary data is transferred:

```
Sender 0 (rows 0-511)    ──▶ chunk [0:512, 0:512]   ──▶ Receiver 0 (cols 0-511)
                         ──▶ chunk [0:512, 512:1024] ──▶ Receiver 1 (cols 512-1023)

Sender 1 (rows 512-1023) ──▶ chunk [512:1024, 0:512]   ──▶ Receiver 0 (cols 0-511)
                         ──▶ chunk [512:1024, 512:1024] ──▶ Receiver 1 (cols 512-1023)
```

**Pros**: Minimal data transfer - only what's needed
**Cons**: More complex overlap computation and handle management

## Implementation

### CUDA IPC Handles

Both approaches use CUDA IPC for zero-copy GPU-to-GPU memory transfer:

1. **Sender** extracts an IPC handle from its tensor's CUDA storage via `storage._share_cuda_()`
2. **Handle** is serialized and sent to receivers (small metadata, not the tensor data)
3. **Receiver** reconstructs a tensor view via `torch.UntypedStorage._new_shared_cuda()`
4. **Copy** transfers data directly between GPU memories over NVLink

### Actor-Based Architecture

The implementation uses [Monarch](https://github.com/pytorch-labs/monarch) actors for distributed coordination:

1. **Sender actors** initialize their DTensor shards and extract IPC handles
2. **One-way message passing** sends handles from senders to receivers (setup phase)
3. **Receivers open handles once** and store references for repeated transfers
4. **Hot path** (`receive()`) only performs the actual `copy_()` operations

This separation of setup vs. transfer minimizes per-iteration overhead.

### Multi-Stream Copies

To maximize NVLink utilization, copies are parallelized across multiple CUDA streams:

```python
streams = [torch.cuda.Stream() for _ in range(n_streams)]
for i, chunk in enumerate(chunks):
    with torch.cuda.stream(streams[i % n_streams]):
        dst[slices].copy_(chunk)
```

### Overlap Detection (DCP Concepts)

The routed approach uses concepts from PyTorch's Distributed Checkpoint (DCP) resharding algorithm to compute which sender regions overlap with which receiver regions:

- **ChunkStorageMetadata**: Describes a shard's position in the global tensor (offsets + sizes)
- **Overlap detection**: For each (sender, receiver) pair, compute the intersection region
- **Transfer plan**: Map each sender chunk to its destination receiver and local coordinates

```python
# Using DCP's overlap utilities
from torch.distributed.checkpoint.metadata import ChunkStorageMetadata
from torch.distributed.checkpoint.resharding import (
    _check_shard_metadata_pair_overlap,
    _shards_get_overlap_region_wrt_saved_tensor,
)
```

## Benchmarks

**Hardware**: NVIDIA H100 80GB GPUs with NVSwitch (NV18 topology - 18 NVLinks per GPU)

**Configuration**: 2 senders (GPUs 0-1) → 2 receivers (GPUs 2-3), 4 CUDA streams

### Gather Approach (10 GB tensor)

| Metric | Value |
|--------|-------|
| Tensor size | 10,000 MB |
| Mean time | 81.5 ms |
| Std dev | 0.4 ms |
| **Throughput** | **119.9 GB/s** |

### Routed Approach (50 GB tensor)

| Metric | Value |
|--------|-------|
| Tensor size | 50,176 MB |
| Mean time | 164.2 ms |
| Std dev | 0.2 ms |
| **Throughput** | **298.4 GB/s** |

### Notes

- Theoretical NVLink bandwidth: **450 GB/s** unidirectional per GPU pair
- Current utilization: ~66% of theoretical peak
- The gap is likely due to strided memory access patterns (column slicing) rather than contiguous transfers

## Usage

### Installation

```bash
uv pip install torchmonarch==0.2.0
```

### Running

```bash
# Gather approach
python ipc_gather.py

# Routed/direct approach
python ipc_routed.py
```

### Configuration

Edit the configuration section in each script:

```python
n_senders = 2
n_receivers = 2
tensor_shape = (51200, 51200)  # 10 GB
n_streams = 4
```
