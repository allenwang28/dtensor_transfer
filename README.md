# DTensor transfer

Basic explorations of transferring data from one DTensor to another across GPUs.

Communication protocols:
- IPC, i.e. CUDA<>CUDA transfers, restricted to the same node
- RDMA, i.e. NIC<>NIC transfers, not restricted to the same node
