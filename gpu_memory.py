from typing import overload


class GpuMemoryBase(object):
    gpu_address: int
    gpu_memory_size: int

    def __init__(self, gpu_address: int, gpu_memory_size: int) -> None:
        self.gpu_address = gpu_address
        self.gpu_memory_size = gpu_memory_size

    @overload
    def __getitem__(self, index: slice) -> bytes:
        return b""

    @overload
    def __setitem__(self, index: slice, object: bytes) -> None:
        pass

    @overload
    def close(self):
        pass

    def __repr__(self) -> str:
        return f"GpuMemoryBase(gpu_address=0x{self.gpu_address:x}, gpu_memory_size=0x{self.gpu_memory_size:x})"
