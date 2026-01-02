# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
RemoteTensor: Unified abstraction for cross-process tensor access.

Supports both CUDA IPC (same-node) and RDMA (cross-node) transports
with a consistent API.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Tuple

import torch


class Transport(Enum):
    """Transport mechanism for remote tensor access."""

    IPC = "ipc"  # CUDA IPC (same node, fastest)
    RDMA = "rdma"  # RDMA (cross-node capable)
    AUTO = "auto"  # Pick best available (prefers IPC)


@dataclass
class RemoteTensor:
    """
    A handle to a tensor owned by another process.

    Supports both IPC and RDMA transports. The handle is serializable
    and can be sent between processes via Monarch actor messages.

    Usage:
        # Sender creates handle
        handle = RemoteTensor.from_tensor(tensor, owner="sender_0")

        # ... handle sent to receiver via actor message ...

        # Receiver uses handle to transfer data
        handle.read_into(local_tensor, transport=Transport.AUTO)
    """

    # Tensor metadata
    shape: Tuple[int, ...]
    stride: Tuple[int, ...]
    dtype: torch.dtype
    owner: str  # actor/process ID

    # IPC handle fields (same-node only)
    ipc_device: Optional[int] = None
    ipc_handle: Optional[Any] = None
    ipc_size_bytes: Optional[int] = None
    ipc_offset_bytes: Optional[int] = None
    ipc_ref_counter_handle: Optional[Any] = None
    ipc_ref_counter_offset: Optional[int] = None
    ipc_event_handle: Optional[Any] = None
    ipc_event_sync_required: Optional[bool] = None

    # RDMA handle (cross-node capable)
    # Type is Any to avoid import dependency; will be RDMABuffer at runtime
    rdma_buffer: Optional[Any] = None

    # Cached opened IPC tensor (not serialized, reopened on remote)
    _ipc_tensor: Optional[torch.Tensor] = field(default=None, repr=False)

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        owner: str,
        *,
        enable_ipc: bool = True,
        enable_rdma: bool = False,
    ) -> "RemoteTensor":
        """
        Create a RemoteTensor handle from a local CUDA tensor.

        Args:
            tensor: The local tensor to expose (must be CUDA for IPC)
            owner: Identifier for this process/actor
            enable_ipc: Create IPC handle (for same-node transfers)
            enable_rdma: Create RDMA handle (for cross-node transfers)

        Returns:
            RemoteTensor handle that can be serialized and sent to other processes
        """
        # Ensure contiguous for IPC
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        handle = cls(
            shape=tuple(tensor.shape),
            stride=tuple(tensor.stride()),
            dtype=tensor.dtype,
            owner=owner,
        )

        if enable_ipc and tensor.is_cuda:
            storage = tensor.untyped_storage()
            (
                handle.ipc_device,
                handle.ipc_handle,
                handle.ipc_size_bytes,
                handle.ipc_offset_bytes,
                handle.ipc_ref_counter_handle,
                handle.ipc_ref_counter_offset,
                handle.ipc_event_handle,
                handle.ipc_event_sync_required,
            ) = storage._share_cuda_()

        if enable_rdma:
            from monarch.rdma import RDMABuffer

            byte_view = tensor.view(torch.uint8).flatten()
            handle.rdma_buffer = RDMABuffer(byte_view)

        return handle

    @property
    def has_ipc(self) -> bool:
        """Whether IPC transport is available."""
        return self.ipc_handle is not None

    @property
    def has_rdma(self) -> bool:
        """Whether RDMA transport is available."""
        return self.rdma_buffer is not None

    @property
    def size_bytes(self) -> int:
        """Total size of the tensor in bytes."""
        return (
            torch.tensor([], dtype=self.dtype).element_size()
            * torch.Size(self.shape).numel()
        )

    def _open_ipc(self) -> torch.Tensor:
        """Lazily open IPC handle and return tensor view."""
        if self._ipc_tensor is None:
            if not self.has_ipc:
                raise RuntimeError("No IPC handle available")
            storage = torch.UntypedStorage._new_shared_cuda(
                self.ipc_device,
                self.ipc_handle,
                self.ipc_size_bytes,
                self.ipc_offset_bytes,
                self.ipc_ref_counter_handle,
                self.ipc_ref_counter_offset,
                self.ipc_event_handle,
                self.ipc_event_sync_required,
            )
            self._ipc_tensor = torch.empty(
                self.shape, dtype=self.dtype, device=f"cuda:{self.ipc_device}"
            )
            self._ipc_tensor.set_(
                storage, storage_offset=0, size=self.shape, stride=self.stride
            )
        return self._ipc_tensor

    def _resolve_transport(self, transport: Transport) -> Transport:
        """Resolve AUTO transport to a concrete transport."""
        if transport == Transport.AUTO:
            # Prefer IPC if available (faster for same-node)
            return Transport.IPC if self.has_ipc else Transport.RDMA
        return transport

    def read_into(
        self,
        dst: torch.Tensor,
        *,
        transport: Transport = Transport.AUTO,
        timeout: int = 3,
    ) -> None:
        """
        Copy data FROM this remote tensor INTO local dst tensor.

        Args:
            dst: Local destination tensor (must match shape/dtype)
            transport: Which transport to use (IPC, RDMA, or AUTO)
            timeout: Timeout in seconds (RDMA only)
        """
        transport = self._resolve_transport(transport)

        if transport == Transport.IPC:
            if not self.has_ipc:
                raise RuntimeError(
                    "IPC transport requested but no IPC handle available"
                )
            src = self._open_ipc()
            dst.copy_(src)

        elif transport == Transport.RDMA:
            if not self.has_rdma:
                raise RuntimeError(
                    "RDMA transport requested but no RDMA handle available"
                )
            byte_view = dst.view(torch.uint8).flatten()
            self.rdma_buffer.read_into(byte_view, timeout=timeout).get()

    def write_from(
        self,
        src: torch.Tensor,
        *,
        transport: Transport = Transport.AUTO,
        timeout: int = 3,
    ) -> None:
        """
        Copy data FROM local src tensor INTO this remote tensor.

        Args:
            src: Local source tensor (must match shape/dtype)
            transport: Which transport to use (IPC, RDMA, or AUTO)
            timeout: Timeout in seconds (RDMA only)
        """
        transport = self._resolve_transport(transport)

        if transport == Transport.IPC:
            if not self.has_ipc:
                raise RuntimeError(
                    "IPC transport requested but no IPC handle available"
                )
            dst = self._open_ipc()
            dst.copy_(src)

        elif transport == Transport.RDMA:
            if not self.has_rdma:
                raise RuntimeError(
                    "RDMA transport requested but no RDMA handle available"
                )
            byte_view = src.view(torch.uint8).flatten()
            self.rdma_buffer.write_from(byte_view, timeout=timeout).get()

    def drop(self) -> None:
        """Release handles and deregister memory."""
        self._ipc_tensor = None
        if self.rdma_buffer is not None:
            self.rdma_buffer.drop().get()
            self.rdma_buffer = None
