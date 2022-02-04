import struct

COMMAND_SUBMISSION_MODE_INCREASING_OLD = 0
COMMAND_SUBMISSION_MODE_INCREASING = 1
COMMAND_SUBMISSION_MODE_NON_INCREASING_OLD = 2
COMMAND_SUBMISSION_MODE_NON_INCREASING = 3
COMMAND_SUBMISSION_MODE_INLINE = 4
COMMAND_SUBMISSION_MODE_INCREASING_ONCE = 5

SUBCHANNEL_ID_3D = 0
SUBCHANNEL_ID_COMPUTE = 1
SUBCHANNEL_ID_I2M = 2
SUBCHANNEL_ID_2D = 3
SUBCHANNEL_ID_DMA = 4

# Internal utils
_METHOD_OFFSET = 0
_METHOD_SIZE = 12

_SUBCHANNEL_OFFSET = 13
_SUBCHANNEL_SIZE = 4

_ARGUMENT_OFFSET = 16
_ARGUMENT_SIZE = 11

# FIXME: bit 28 is probably related to interruption modeling

_SUBMISSION_MODE_OFFSET = 29
_SUBMISSION_MODE_SIZE = 2

def _set_bits(offset: int, size: int, value: int) -> int:
    return ((value & ((1 << (size + 1)) - 1)) << offset)

def InlineCommand(method: int, subchannel: int, argument: int) -> int:
    return Command(method, subchannel, argument, COMMAND_SUBMISSION_MODE_INLINE)

def Command(method: int, subchannel: int, argument: int, submission_mode: int) -> int:
    value = 0
    value |= _set_bits(_METHOD_OFFSET, _METHOD_SIZE, method)
    value |= _set_bits(_SUBCHANNEL_OFFSET, _SUBCHANNEL_SIZE, subchannel)
    value |= _set_bits(_ARGUMENT_OFFSET, _ARGUMENT_SIZE, argument)
    value |= _set_bits(_SUBMISSION_MODE_OFFSET, _SUBMISSION_MODE_SIZE, submission_mode)

    return value

# Setup utils
BIND_CHANNEL_3D = Command(0x0, SUBCHANNEL_ID_3D, 1, COMMAND_SUBMISSION_MODE_INCREASING)
BIND_CHANNEL_COMPUTE = Command(0x0, SUBCHANNEL_ID_COMPUTE, 1, COMMAND_SUBMISSION_MODE_INCREASING)
BIND_CHANNEL_I2M = Command(0x0, SUBCHANNEL_ID_I2M, 1, COMMAND_SUBMISSION_MODE_INCREASING)
BIND_CHANNEL_2D = Command(0x0, SUBCHANNEL_ID_2D, 1, COMMAND_SUBMISSION_MODE_INCREASING)
BIND_CHANNEL_DMA = Command(0x0, SUBCHANNEL_ID_DMA, 1, COMMAND_SUBMISSION_MODE_INCREASING)


class CommandBuffer(object):
    buffer: bytearray

    def __init__(self) -> None:
        self.buffer = bytearray()

    def write_u32(self, data: int) -> None:
        self.write_bytes(struct.pack("I", data))

    def write_u64(self, data: int) -> None:
        self.write_bytes(struct.pack("Q", data))

    def write_bytes(self, data: bytes) -> None:
        self.buffer += bytearray(data)
