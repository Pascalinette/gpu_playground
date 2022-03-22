from command_buffer import *
from nvgpu import GpuMemory, align_up
from maxwell.hw import *
from maxwell.hw.dma_copy_a import *

import typing

MAX_LINEAR_COPY_SIZE = 0x3FFFFF


def memcpy_device_to_device(
    command_buffer: CommandBuffer, dest_address: int, src_address: int, size: int
):
    command_buffer.write_u32(
        InlineCommand(NVB0B5_NOP, SUBCHANNEL_ID_DMA, NVB0B5_NOP_PARAMETER(0))
    )

    while size > 0:
        copy_size = min(size, MAX_LINEAR_COPY_SIZE)

        command_buffer.write_u32(
            IncrCommand(NVB0B5_OFFSET_IN_UPPER, SUBCHANNEL_ID_DMA, 4)
        )
        command_buffer.write_u64(src_address)
        command_buffer.write_u64(dest_address)

        command_buffer.write_u32(
            IncrCommand(NVB0B5_LINE_LENGTH_IN, SUBCHANNEL_ID_DMA, 1)
        )
        command_buffer.write_u32(NVB0B5_LINE_LENGTH_IN_VALUE(copy_size))

        command_buffer.write_u32(
            IncrCommand(NVB0B5_SET_DST_ORIGIN, SUBCHANNEL_ID_DMA, 1)
        )

        launch_dma_value = (
            NVB0B5_LAUNCH_DMA_DATA_TRANSFER_TYPE(
                NVB0B5_LAUNCH_DMA_DATA_TRANSFER_TYPE_NON_PIPELINED
            )
            | NVB0B5_LAUNCH_DMA_FLUSH_ENABLE(NVB0B5_LAUNCH_DMA_FLUSH_ENABLE_TRUE)
            | NVB0B5_LAUNCH_DMA_SRC_MEMORY_LAYOUT(
                NVB0B5_LAUNCH_DMA_SRC_MEMORY_LAYOUT_PITCH
            )
            | NVB0B5_LAUNCH_DMA_SRC_TYPE(NVB0B5_LAUNCH_DMA_SRC_TYPE_VIRTUAL)
            | NVB0B5_LAUNCH_DMA_DST_MEMORY_LAYOUT(
                NVB0B5_LAUNCH_DMA_DST_MEMORY_LAYOUT_PITCH
            )
            | NVB0B5_LAUNCH_DMA_DST_TYPE(NVB0B5_LAUNCH_DMA_DST_TYPE_VIRTUAL)
        )
        command_buffer.write_u32(
            InlineCommand(NVB0B5_LAUNCH_DMA, SUBCHANNEL_ID_DMA, launch_dma_value)
        )

        src_address += copy_size
        dest_address += copy_size
        size -= copy_size

    command_buffer.write_u32(
        InlineCommand(NVB0B5_NOP, SUBCHANNEL_ID_DMA, NVB0B5_NOP_PARAMETER(0))
    )


class ImageInfo(object):
    gpu_address: int

    horizontal: int
    vertical: int
    depth: int

    #format: int
    tile_mode: int
    #array_mode: int
    layer_stride: int

    #width: int
    #height: int
    width_ms: int
    height_ms: int

    bytes_per_block: int
    is_linear: bool
    is_layered: bool

    def __init__(self) -> None:
        pass


class BlitParameters(object):
    src_x: int
    src_y: int
    src_z: int
    dst_x: int
    dst_y: int
    dst_z: int

    width: int
    height: int

    def __init__(
        self,
        src_x: int,
        src_y: int,
        src_z: int,
        dst_x: int,
        dst_y: int,
        dst_z: int,
        width: int,
        height: int,
    ) -> None:
        self.src_x = src_x
        self.src_y = src_y
        self.src_z = src_z
        self.dst_x = dst_x
        self.dst_y = dst_y
        self.dst_z = dst_z
        self.width = width
        self.height = height


def blit(
    command_buffer: CommandBuffer,
    dest_image: ImageInfo,
    src_image: ImageInfo,
    parameters: BlitParameters,
    should_miss_origin: bool = False
):
    launch_dma_flags = (
        NVB0B5_LAUNCH_DMA_DATA_TRANSFER_TYPE(
            NVB0B5_LAUNCH_DMA_DATA_TRANSFER_TYPE_NON_PIPELINED
        )
        | NVB0B5_LAUNCH_DMA_FLUSH_ENABLE(NVB0B5_LAUNCH_DMA_FLUSH_ENABLE_TRUE)
        | NVB0B5_LAUNCH_DMA_MULTI_LINE_ENABLE(NVB0B5_LAUNCH_DMA_MULTI_LINE_ENABLE_TRUE)
        | NVB0B5_LAUNCH_DMA_SRC_TYPE(NVB0B5_LAUNCH_DMA_SRC_TYPE_VIRTUAL)
        | NVB0B5_LAUNCH_DMA_DST_TYPE(NVB0B5_LAUNCH_DMA_DST_TYPE_VIRTUAL)
    )

    dest_address = dest_image.gpu_address
    src_address = src_image.gpu_address
    use_swizzle = False

    if dest_image.is_layered:
        dest_address += parameters.dst_z * dest_image.layer_stride
    if dest_image.is_linear:
        dest_address += (
            parameters.dst_y * dest_image.horizontal
            + parameters.dst_x * dest_image.bytes_per_block
        )
    elif dest_image.width_ms * dest_image.bytes_per_block > 0x10000:
        use_swizzle = True

    if src_image.is_layered:
        src_address += parameters.src_z * src_image.layer_stride
    if src_image.is_linear:
        src_address += (
            parameters.src_y * src_image.horizontal
            + parameters.src_x * src_image.bytes_per_block
        )
    elif src_image.width_ms * src_image.bytes_per_block > 0x10000:
        use_swizzle = True

    src_horiz_factor = 1
    dst_horiz_factor = 1
    src_x = parameters.src_x
    dst_x = parameters.dst_x
    width = parameters.width

    if not use_swizzle:
        src_x *= src_image.bytes_per_block
        dst_x *= dest_image.bytes_per_block
        width *= src_image.bytes_per_block
        src_horiz_factor = src_image.bytes_per_block
        dst_horiz_factor = dest_image.bytes_per_block

    if not dest_image.is_linear:
        target_command_size = 6
        if should_miss_origin:
            target_command_size -= 1

        command_buffer.write_u32(
            IncrCommand(NVB0B5_SET_DST_BLOCK_SIZE, SUBCHANNEL_ID_DMA, target_command_size)
        )
        command_buffer.write_u32(
            src_image.tile_mode
            | NVB0B5_SET_DST_BLOCK_SIZE_GOB_HEIGHT(
                NVB0B5_SET_DST_BLOCK_SIZE_GOB_HEIGHT_GOB_HEIGHT_FERMI_8
            )
        )

        command_buffer.write_u32(
            NVB0B5_SET_DST_WIDTH_V(dest_image.horizontal * dst_horiz_factor)
        )
        command_buffer.write_u32(NVB0B5_SET_DST_HEIGHT_V(dest_image.vertical))
        command_buffer.write_u32(NVB0B5_SET_DST_DEPTH_V(dest_image.depth))
        command_buffer.write_u32(
            NVB0B5_SET_DST_LAYER_V(0 if not dest_image.is_layered else parameters.dst_z)
        )

        if not should_miss_origin:
            command_buffer.write_u32(
                NVB0B5_SET_DST_ORIGIN_X(parameters.dst_x)
                | NVB0B5_SET_DST_ORIGIN_Y(parameters.dst_y)
            )
    else:
        launch_dma_flags |= NVB0B5_LAUNCH_DMA_DST_MEMORY_LAYOUT(
            NVB0B5_LAUNCH_DMA_DST_MEMORY_LAYOUT_PITCH
        )

        command_buffer.write_u32(IncrCommand(NVB0B5_PITCH_OUT, SUBCHANNEL_ID_DMA, 1))
        command_buffer.write_u32(NVB0B5_PITCH_IN_VALUE(dest_image.horizontal))

    if not src_image.is_linear:
        command_buffer.write_u32(
            IncrCommand(NVB0B5_SET_SRC_BLOCK_SIZE, SUBCHANNEL_ID_DMA, 6)
        )
        command_buffer.write_u32(
            src_image.tile_mode
            | NVB0B5_SET_SRC_BLOCK_SIZE_GOB_HEIGHT(
                NVB0B5_SET_SRC_BLOCK_SIZE_GOB_HEIGHT_GOB_HEIGHT_FERMI_8
            )
        )

        command_buffer.write_u32(
            NVB0B5_SET_SRC_WIDTH_V(src_image.horizontal * src_horiz_factor)
        )
        command_buffer.write_u32(NVB0B5_SET_SRC_HEIGHT_V(src_image.vertical))
        command_buffer.write_u32(NVB0B5_SET_SRC_DEPTH_V(src_image.depth))
        command_buffer.write_u32(
            NVB0B5_SET_SRC_LAYER_V(0 if not src_image.is_layered else parameters.src_z)
        )
        command_buffer.write_u32(
            NVB0B5_SET_SRC_ORIGIN_X(parameters.src_x)
            | NVB0B5_SET_SRC_ORIGIN_Y(parameters.src_y)
        )
    else:
        launch_dma_flags |= NVB0B5_LAUNCH_DMA_SRC_MEMORY_LAYOUT(
            NVB0B5_LAUNCH_DMA_SRC_MEMORY_LAYOUT_PITCH
        )

        command_buffer.write_u32(IncrCommand(NVB0B5_PITCH_IN, SUBCHANNEL_ID_DMA, 1))
        command_buffer.write_u32(NVB0B5_PITCH_IN_VALUE(src_image.horizontal))

    if use_swizzle:
        launch_dma_flags |= NVB0B5_LAUNCH_DMA_REMAP_ENABLE(
            NVB0B5_LAUNCH_DMA_REMAP_ENABLE_TRUE
        )

        component_size = NVB0B5_SET_REMAP_COMPONENTS_COMPONENT_SIZE_TWO

        if src_image.bytes_per_block in [4, 8, 16]:
            component_size = NVB0B5_SET_REMAP_COMPONENTS_COMPONENT_SIZE_FOUR

            command_buffer.write_u32(
                IncrCommand(NVB0B5_SET_REMAP_CONST_A, SUBCHANNEL_ID_DMA, 3)
            )
            command_buffer.write_u32(NVB0B5_SET_REMAP_CONST_A_V(0))
            command_buffer.write_u32(NVB0B5_SET_REMAP_CONST_B_V(0))

            remap_components_flags = (
                NVB0B5_SET_REMAP_COMPONENTS_DST_X(
                    NVB0B5_SET_REMAP_COMPONENTS_DST_X_SRC_X
                )
                | NVB0B5_SET_REMAP_COMPONENTS_DST_Y(
                    NVB0B5_SET_REMAP_COMPONENTS_DST_Y_SRC_Y
                )
                | NVB0B5_SET_REMAP_COMPONENTS_DST_Z(
                    NVB0B5_SET_REMAP_COMPONENTS_DST_Z_SRC_Z
                )
                | NVB0B5_SET_REMAP_COMPONENTS_DST_W(
                    NVB0B5_SET_REMAP_COMPONENTS_DST_W_SRC_W
                )
                | NVB0B5_SET_REMAP_COMPONENTS_COMPONENT_SIZE(component_size)
                | NVB0B5_SET_REMAP_COMPONENTS_NUM_SRC_COMPONENTS(
                    src_image.bytes_per_block / (component_size + 1) - 1
                )
                | NVB0B5_SET_REMAP_COMPONENTS_NUM_DST_COMPONENTS(
                    dest_image.bytes_per_block / (component_size + 1) - 1
                )
            )

            command_buffer.write_u32(remap_components_flags)

    command_buffer.write_u32(IncrCommand(NVB0B5_OFFSET_IN_UPPER, SUBCHANNEL_ID_DMA, 4))
    command_buffer.write_u64(src_address)
    command_buffer.write_u64(dest_address)

    command_buffer.write_u32(IncrCommand(NVB0B5_LINE_LENGTH_IN, SUBCHANNEL_ID_DMA, 2))
    command_buffer.write_u32(width)
    command_buffer.write_u32(parameters.height)

    command_buffer.write_u32(IncrCommand(NVB0B5_LAUNCH_DMA, SUBCHANNEL_ID_DMA, 1))
    command_buffer.write_u32(launch_dma_flags)
