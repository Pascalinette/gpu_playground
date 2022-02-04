import struct
from utils import set_bits
from typing import Union

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

def InlineCommand(method: int, subchannel: int, argument: int, is_raw_method: bool = False) -> int:
    return Command(method, subchannel, argument, COMMAND_SUBMISSION_MODE_INLINE, is_raw_method)

def IncrCommand(method: int, subchannel: int, argument: int, is_raw_method: bool = False) -> int:
    return Command(method, subchannel, argument, COMMAND_SUBMISSION_MODE_INCREASING, is_raw_method)

def NonIncrCommand(method: int, subchannel: int, argument: int, is_raw_method: bool = False) -> int:
    return Command(method, subchannel, argument, COMMAND_SUBMISSION_MODE_NON_INCREASING, is_raw_method)

def Command(method: int, subchannel: int, argument: int, submission_mode: int, is_raw_method: bool = False) -> int:
    if not is_raw_method:
        method //= 4

    value = 0
    value |= set_bits(_METHOD_OFFSET, _METHOD_SIZE, method)
    value |= set_bits(_SUBCHANNEL_OFFSET, _SUBCHANNEL_SIZE, subchannel)
    value |= set_bits(_ARGUMENT_OFFSET, _ARGUMENT_SIZE, argument)
    value |= set_bits(_SUBMISSION_MODE_OFFSET, _SUBMISSION_MODE_SIZE, submission_mode)

    return value

# Setup utils
BIND_CHANNEL_3D = IncrCommand(0x0, SUBCHANNEL_ID_3D, 1)
BIND_CHANNEL_COMPUTE = IncrCommand(0x0, SUBCHANNEL_ID_COMPUTE, 1)
BIND_CHANNEL_I2M = IncrCommand(0x0, SUBCHANNEL_ID_I2M, 1)
BIND_CHANNEL_2D = IncrCommand(0x0, SUBCHANNEL_ID_2D, 1)
BIND_CHANNEL_DMA = IncrCommand(0x0, SUBCHANNEL_ID_DMA, 1)


class CommandBuffer(object):
    buffer: bytearray

    def __init__(self) -> None:
        self.buffer = bytearray()

    def write_u32(self, data: int) -> None:
        self.write_bytes(struct.pack("I", data))

    def write_u64(self, data: int) -> None:
        self.write_u32((data >> 32) & 0xFFFFFFFF)
        self.write_u32(data & 0xFFFFFFFF)

    def write_bytes(self, data: Union[bytes, bytearray]) -> None:
        self.buffer += bytearray(data)
