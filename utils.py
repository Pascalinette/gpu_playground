from ctypes import CDLL, addressof, c_int, POINTER, c_ubyte, memmove, sizeof
import errno
from os import close
import os
from typing import Any, Optional


libc = CDLL("libc.so.6")
get_errno_loc = libc.__errno_location
get_errno_loc.restype = POINTER(c_int)


_IOC_NRBITS = 8
_IOC_TYPEBITS = 8
_IOC_NRSHIFT = 0

# Arch specific (TODO: put correct value between archs and make this user configurable)
_IOC_SIZEBITS = 14
_IOC_DIRBITS = 2
_IOC_NONE = 0
_IOC_WRITE = 1
_IOC_READ = 2

_IOC_NRMASK = (1 << _IOC_NRBITS) - 1
_IOC_TYPEMASK = (1 << _IOC_TYPEBITS) - 1
_IOC_SIZEMASK = (1 << _IOC_SIZEBITS) - 1
_IOC_DIRMASK = (1 << _IOC_DIRBITS) - 1


_IOC_TYPESHIFT = _IOC_NRSHIFT + _IOC_NRBITS
_IOC_SIZESHIFT = _IOC_TYPESHIFT + _IOC_TYPEBITS
_IOC_DIRSHIFT = _IOC_SIZESHIFT + _IOC_SIZEBITS


def IOC(dir: int, type: int, nr: int, size: int) -> int:
    assert dir <= _IOC_DIRMASK
    assert type <= _IOC_TYPEMASK
    assert nr <= _IOC_NRMASK
    assert size <= _IOC_SIZEMASK

    return (
        (dir << _IOC_DIRSHIFT)
        | (type << _IOC_TYPESHIFT)
        | (nr << _IOC_NRSHIFT)
        | (size << _IOC_SIZESHIFT)
    )


# NOTE: _CData in ctypes is private so t cannot be represented.
def IOC_TYPECHECK(t: Any) -> int:
    result = sizeof(t)

    assert result <= _IOC_SIZEMASK

    return sizeof(t)


def IO(type: int, nr: int) -> int:
    return IOC(_IOC_NONE, type, nr, 0)


def IOR(type: int, nr: int, size: Any) -> int:
    return IOC(_IOC_READ, type, nr, IOC_TYPECHECK(size))


def IOW(type: int, nr: int, size: Any) -> int:
    return IOC(_IOC_WRITE, type, nr, IOC_TYPECHECK(size))


def IOWR(type: int, nr: int, size: Any) -> int:
    return IOC(_IOC_READ | _IOC_WRITE, type, nr, IOC_TYPECHECK(size))


def IOR_BAD(type: int, nr: int, size: Any) -> int:
    return IOC(_IOC_READ, type, nr, sizeof(size))


def IOW_BAD(type: int, nr: int, size: Any) -> int:
    return IOC(_IOC_WRITE, type, nr, sizeof(size))


def IORW_BAD(type: int, nr: int, size: Any) -> int:
    return IOC(_IOC_READ | _IOC_WRITE, type, nr, sizeof(size))


def IOC_DIR(nr: int) -> int:
    return (nr >> _IOC_DIRSHIFT) & _IOC_DIRMASK


def IOC_TYPE(nr: int) -> int:
    return (nr >> _IOC_TYPESHIFT) & _IOC_TYPEMASK


def IOC_NR(nr: int) -> int:
    return (nr >> _IOC_NRSHIFT) & _IOC_NRMASK


def IOC_SIZE(nr: int) -> int:
    return (nr >> _IOC_SIZESHIFT) & _IOC_SIZEMASK


def set_bits(offset: int, size: int, value: int) -> int:
    return (value & ((1 << (size + 1)) - 1)) << offset


def align_up(value: int, size: int) -> int:
    return (value + (size - 1)) & -size


def align_down(value: int, size: int) -> int:
    return value & -size


def get_errno() -> int:
    return get_errno_loc()[0]


class ErrnoException(Exception):
    """Raised when an errno is returned"""

    error: int
    message: Optional[str]

    def __init__(self, error: int) -> None:
        self.error = error

        if error in errno.errorcode:
            self.message = errno.errorcode[error]
        else:
            self.message = f"Unknown errno {error}"

        super().__init__(error)


def get_errno_code() -> Optional[str]:
    error = get_errno()

    if error != 0:
        return errno.errorcode[error]

    return None


def check_result(result_code: int, output_value: Any = None) -> Any:
    if result_code >= 0:
        return output_value

    exception = ErrnoException(get_errno())
    print(exception.message)

    raise exception


class BlockDevice(object):
    fd: int

    def __init__(self, path: Optional[str] = None, fd: int = -1) -> None:
        if fd != -1:
            self.fd = fd
        elif path is not None:
            self.fd = os.open(path, os.O_RDWR | os.O_CLOEXEC)
        else:
            raise Exception("INVALID COMBINAISON")

    def close(self) -> None:
        close(self.fd)

    def check_result(self, result_code: int, output_value: Any = None) -> Any:
        return check_result(result_code, output_value)


def ctypes_struct_to_bytearray(struct: Any) -> bytearray:
    raw_buff = (c_ubyte * sizeof(struct))()

    memmove(raw_buff, addressof(struct), sizeof(struct))

    return bytearray(raw_buff)


def bytearray_to_ctypes_struct(struct: Any, buff: bytearray):
    raw_buff = (c_ubyte * sizeof(struct)).from_buffer(buff)

    memmove(addressof(struct), raw_buff, sizeof(struct))
