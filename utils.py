from ctypes import sizeof
from typing import Any

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
