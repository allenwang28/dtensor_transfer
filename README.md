# DTensor Transfer

Exploring how to efficiently transfer data between DTensors with different sharding layouts across GPUs.

## Overview

When working with distributed tensors (DTensors), a common operation is **resharding** - changing how a tensor is partitioned across devices. For example, transitioning from row-wise sharding (`Shard(0)`) to column-wise sharding (`Shard(1)`) between different model parallelism configurations.

## Approaches

### Gather (`gather.py`)

Each receiver gathers **all** sender shards into a buffer, then extracts its column portion:

```
Sender 0 (rows 0-511)    ──┐
                           ├──▶ Receiver 0: gather all → slice cols 0-511
Sender 1 (rows 512-1023) ──┘

Sender 0 (rows 0-511)    ──┐
                           ├──▶ Receiver 1: gather all → slice cols 512-1023
Sender 1 (rows 512-1023) ──┘
```

**Pros**: Simple implementation
**Cons**: Each receiver transfers the entire tensor, even though it only needs half

### Routed / Direct (`routed.py`)

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

### RemoteTensor (`remote_tensor.py`)

The `RemoteTensor` class provides a unified abstraction for cross-process tensor access, supporting both **CUDA IPC** (same-node) and **RDMA** (cross-node) transports:

```python
from remote_tensor import RemoteTensor, Transport

# Sender: create a handle from a local tensor
handle = RemoteTensor.from_tensor(
    tensor,
    owner="sender_0",
    enable_ipc=True,   # Same-node via CUDA IPC
    enable_rdma=False, # Cross-node via RDMA
)

# ... handle is serialized and sent to receiver via actor message ...

# Receiver: copy data into local tensor
handle.read_into(local_tensor, transport=Transport.AUTO)

# Or write to the remote tensor
handle.write_from(local_tensor, transport=Transport.IPC)
```

**Key features:**
- **Transport selection at call time**: Choose `Transport.IPC`, `Transport.RDMA`, or `Transport.AUTO` (prefers IPC)
- **Lazy handle opening**: IPC handles are opened once and cached for repeated transfers
- **Serializable**: Handles can be passed between processes via Monarch actor messages
- **Unified API**: Same `read_into()`/`write_from()` interface for both transports

**Transport comparison:**

| Transport | Use Case | Mechanism |
|-----------|----------|-----------|
| IPC | Same-node GPU transfers | `storage._share_cuda_()` / `UntypedStorage._new_shared_cuda()` |
| RDMA | Cross-node transfers | Monarch `RDMABuffer` with `read_into()` / `write_from()` |

### Actor-Based Architecture

The implementation uses [Monarch](https://github.com/pytorch-labs/monarch) actors for distributed coordination:

1. **Sender actors** initialize their DTensor shards and create RemoteTensor handles
2. **One-way message passing** sends handles from senders to receivers (setup phase)
3. **Receivers store handles** and call `read_into()` for repeated transfers
4. **Hot path** (`receive()`) only performs the actual data transfers

This separation of setup vs. transfer minimizes per-iteration overhead.

### Multi-Stream Copies

To maximize NVLink utilization, copies are parallelized across multiple CUDA streams:

```python
streams = [torch.cuda.Stream() for _ in range(n_streams)]
for i, (rt, slices) in enumerate(chunk_info):
    with torch.cuda.stream(streams[i % n_streams]):
        rt.read_into(local_tensor[slices], transport=Transport.AUTO)
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

### Routed Approach (10 GB tensor)

| Metric | Value |
|--------|-------|
| Tensor size | 10,000 MB |
| Mean time | 35.4 ms |
| Std dev | 0.1 ms |
| **Throughput** | **~280 GB/s** |

### Notes

- Theoretical NVLink bandwidth: **450 GB/s** unidirectional per GPU pair
- Current utilization: ~60-65% of theoretical peak
- The gap is likely due to strided memory access patterns (column slicing) rather than contiguous transfers

## Usage

### Installation

```bash
uv pip install torchmonarch==0.2.0
```

### Running

```bash
# Gather approach
python gather.py

# Routed/direct approach
python routed.py
```

### Configuration

Edit the configuration section in each script:

```python
n_senders = 2
n_receivers = 2
tensor_shape = (51200, 51200)  # 10 GB
n_streams = 4
transport = "auto"  # "ipc", "rdma", or "auto"
```
