from command_buffer import *
from utils import set_bits
from maxwell.hw import *
from maxwell.hw.compute_b import *

# TODO: Parse NVIDIA bitfields from open-gpu docs or maybe define our own register definition format to simplify this mess
NVB1C0_LAUNCH_DMA_DST_MEMORY_LAYOUT_OFFSET = 0
NVB1C0_LAUNCH_DMA_DST_MEMORY_LAYOUT_SIZE = 0

NVB1C0_LAUNCH_DMA_COMPLETION_TYPE_OFFSET = 4
NVB1C0_LAUNCH_DMA_COMPLETION_TYPE_SIZE = 1


def memcpy_inline_host_to_device(
    command_buffer: CommandBuffer, dest_address: int, data: bytes
):
    buffer = bytearray(data)

    while len(buffer) % 4 != 0:
        buffer += bytearray(b"\x00")

    command_buffer.write_u32(
        IncrCommand(NVB1C0_LINE_LENGTH_IN, SUBCHANNEL_ID_COMPUTE, 4)
    )
    command_buffer.write_u32(len(data))
    command_buffer.write_u32(1)
    command_buffer.write_u64(dest_address)

    command_buffer.write_u32(IncrCommand(NVB1C0_LAUNCH_DMA, SUBCHANNEL_ID_COMPUTE, 1))

    launch_dma_value = set_bits(
        NVB1C0_LAUNCH_DMA_DST_MEMORY_LAYOUT_OFFSET,
        NVB1C0_LAUNCH_DMA_DST_MEMORY_LAYOUT_SIZE,
        NVB1C0_LAUNCH_DMA_DST_MEMORY_LAYOUT_PITCH,
    ) | set_bits(
        NVB1C0_LAUNCH_DMA_COMPLETION_TYPE_OFFSET,
        NVB1C0_LAUNCH_DMA_COMPLETION_TYPE_SIZE,
        NVB1C0_LAUNCH_DMA_COMPLETION_TYPE_FLUSH_ONLY,
    )

    command_buffer.write_u32(launch_dma_value)

    command_buffer.write_u32(
        NonIncrCommand(NVB1C0_LOAD_INLINE_DATA, SUBCHANNEL_ID_COMPUTE, len(buffer) // 4)
    )
    command_buffer.write_bytes(buffer)
