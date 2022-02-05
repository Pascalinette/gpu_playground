from command_buffer import *
from nvgpu import GpuMemory, align_up
from maxwell.hw import *
from maxwell.hw.compute_b import *
from maxwell.hw.compute_b_qmd import *

import typing

CONSTANT_BUFFER_MAX_SIZE = 0x10000
CONSTANT_BUFFER_ALIGN_REQUIREMENT = 0x100


def initialize_compute_engine(
    command_buffer: CommandBuffer,
    scratch_memory: GpuMemory,
    shader_program_memory: GpuMemory,
    bindless_texture_cbuff_index: int,
    sm_count: int,
):
    command_buffer.write_u32(
        InlineCommand(NVB1C0_SET_SHADER_EXCEPTIONS, SUBCHANNEL_ID_COMPUTE, 0)
    )

    command_buffer.write_u32(
        InlineCommand(
            NVB1C0_SET_BINDLESS_TEXTURE,
            SUBCHANNEL_ID_COMPUTE,
            bindless_texture_cbuff_index,
        )
    )

    command_buffer.write_u32(
        IncrCommand(NVB1C0_SET_SHADER_LOCAL_MEMORY_WINDOW, SUBCHANNEL_ID_COMPUTE, 1)
    )
    command_buffer.write_u32(0x1000000)

    command_buffer.write_u32(
        IncrCommand(NVB1C0_SET_SHADER_SHARED_MEMORY_WINDOW, SUBCHANNEL_ID_COMPUTE, 1)
    )
    command_buffer.write_u32(0x3000000)

    command_buffer.write_u32(
        IncrCommand(NVB1C0_SET_PROGRAM_REGION_A, SUBCHANNEL_ID_COMPUTE, 2)
    )
    command_buffer.write_u64(shader_program_memory.gpu_address)

    command_buffer.write_u32(
        InlineCommand(
            NVB1C0_SET_SPA_VERSION,
            SUBCHANNEL_ID_COMPUTE,
            NVB1C0_SET_SPA_VERSION_MAJOR(5) | NVB1C0_SET_SPA_VERSION_MAJOR(3),
        )
    )

    command_buffer.write_u32(
        IncrCommand(NVB1C0_SET_SHADER_LOCAL_MEMORY_A, SUBCHANNEL_ID_COMPUTE, 2)
    )
    command_buffer.write_u64(scratch_memory.gpu_address)

    scratch_memory_per_sm = scratch_memory.gpu_memory_size // sm_count
    command_buffer.write_u32(
        IncrCommand(
            NVB1C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A, SUBCHANNEL_ID_COMPUTE, 6
        )
    )
    command_buffer.write_u32(0)
    command_buffer.write_u32(scratch_memory_per_sm)
    command_buffer.write_u32(0x100)
    command_buffer.write_u32(0)
    command_buffer.write_u32(scratch_memory_per_sm)
    command_buffer.write_u32(0x100)


def memcpy_inline_host_to_device(
    command_buffer: CommandBuffer,
    dest_address: int,
    data: typing.Union[bytes, bytearray],
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

    launch_dma_value = NVB1C0_LAUNCH_DMA_DST_MEMORY_LAYOUT(
        NVB1C0_LAUNCH_DMA_DST_MEMORY_LAYOUT_PITCH
    ) | NVB1C0_LAUNCH_DMA_COMPLETION_TYPE(NVB1C0_LAUNCH_DMA_COMPLETION_TYPE_FLUSH_ONLY)

    command_buffer.write_u32(launch_dma_value)

    command_buffer.write_u32(
        NonIncrCommand(NVB1C0_LOAD_INLINE_DATA, SUBCHANNEL_ID_COMPUTE, len(buffer) // 4)
    )
    command_buffer.write_bytes(buffer)


def _init_qmd_common(
    job: typing.Union[qmdv0006, qmdv0107], major_version: int, minor_version: int
):
    # NOTE: API_VISIBLE_CALL and L1_CONFIGURATION are the same so we use one of the QMD variant constants.
    job.qmd_major_version = major_version
    job.qmd_version = minor_version
    job.api_visible_call_limit = NVB1C0_QMDV01_07_API_VISIBLE_CALL_LIMIT_NO_CHECK
    job.sm_global_caching_enable = 1
    job.l1_configuration = (
        NVB1C0_QMDV01_07_L1_CONFIGURATION_DIRECTLY_ADDRESSABLE_MEMORY_SIZE_48KB
    )


def create_qmdv0006() -> qmdv0006:
    job = qmdv0006()
    _init_qmd_common(job, 0, 6)

    return job


def create_qmdv0107() -> qmdv0107:
    job = qmdv0107()
    _init_qmd_common(job, 1, 7)

    return job


def qmd_bind_constant_buffer(
    job: typing.Union[qmdv0006, qmdv0107], index, address: int, size: int
):
    if address == 0 or size == 0:
        setattr(job, f"constant_buffer_valid_{index}", 0)
    else:
        assert align_up(address, CONSTANT_BUFFER_ALIGN_REQUIREMENT) == address
        assert align_up(size, CONSTANT_BUFFER_ALIGN_REQUIREMENT) == size
        assert size <= CONSTANT_BUFFER_MAX_SIZE

        setattr(job, f"constant_buffer_valid_{index}", 1)
        setattr(job, f"constant_buffer_addr_lower_{index}", address & 0xFFFFFFFF)
        setattr(job, f"constant_buffer_addr_upper_{index}", (address >> 32) & 0xFF)
        setattr(job, f"constant_buffer_size_{index}", size)


def execute_job(command_buffer: CommandBuffer, job_address: int, job: bytearray):
    memcpy_inline_host_to_device(command_buffer, job_address, job)

    command_buffer.write_u32(IncrCommand(NVB1C0_SEND_PCAS_A, SUBCHANNEL_ID_COMPUTE, 1))
    command_buffer.write_u32(NVB1C0_SEND_PCAS_A_QMD_ADDRESS_SHIFTED8(job_address >> 8))

    command_buffer.write_u32(
        InlineCommand(
            NVB1C0_SEND_SIGNALING_PCAS_B,
            SUBCHANNEL_ID_COMPUTE,
            NVB1C0_SEND_SIGNALING_PCAS_B_INVALIDATE(
                NVB1C0_SEND_SIGNALING_PCAS_B_INVALIDATE_TRUE
            )
            | NVB1C0_SEND_SIGNALING_PCAS_B_SCHEDULE(
                NVB1C0_SEND_SIGNALING_PCAS_B_SCHEDULE_TRUE
            ),
        )
    )
