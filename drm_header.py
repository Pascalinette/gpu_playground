from ctypes import (
    CDLL,
    Structure,
    c_char,
    c_uint32,
    c_int,
    c_size_t,
    POINTER,
    get_errno,
    pointer,
)
from typing import Any, Optional
from utils import _IOC_NONE, _IOC_READ, _IOC_WRITE, IOC, IOC_TYPECHECK, IOWR

import errno

libc = CDLL("libc.so.6")
ioctl = libc.ioctl

DRM_IOCTL_BASE = 100
DRM_COMMAND_BASE = 0x40


class drm_version(Structure):
    _fields_ = [
        ("version_major", c_int),
        ("version_minor", c_int),
        ("version_patchlevel", c_int),
        ("name_len", c_size_t),
        ("name", POINTER(c_char)),
        ("date_len", c_size_t),
        ("date", POINTER(c_char)),
        ("desc_len", c_size_t),
        ("desc", POINTER(c_char)),
    ]


class drm_gem_close_req(Structure):
    _fields_ = [
        ("handle", c_uint32),
        ("pad", c_uint32),
    ]


DRM_IOCTL_VERSION = IOWR(DRM_IOCTL_BASE, 0x00, drm_version)
DRM_IOCTL_GEM_CLOSE = IOWR(DRM_IOCTL_BASE, 0x09, drm_gem_close_req)


def drm_ioctl(fd: Any, request: int, arg: Any) -> int:
    res = None

    while res is None or (res == -1 and get_errno() in [errno.EINTR, errno.EAGAIN]):
        if arg is not None:
            res = ioctl(fd, request, pointer(arg))
        else:
            res = ioctl(fd, request, None)

    return res


def drm_command(
    fd: Any, dir: int, command_index: int, arg: Any, size: Optional[int]
) -> int:
    if size is None:
        size = IOC_TYPECHECK(arg)

    return drm_ioctl(
        fd, IOC(dir, DRM_IOCTL_BASE, DRM_COMMAND_BASE + command_index, size), arg
    )


def drm_command_none(fd: Any, command_index: int) -> int:
    return drm_command(fd, _IOC_NONE, command_index, None, 0)


def drm_command_read(
    fd: Any, command_index: int, arg: Any, size: Optional[int] = None
) -> int:
    return drm_command(fd, _IOC_READ, command_index, arg, size)


def drm_command_write(
    fd: Any, command_index: int, arg: Any, size: Optional[int] = None
) -> int:
    return drm_command(fd, _IOC_WRITE, command_index, arg, size)


def drm_command_read_write(
    fd: Any, command_index: int, arg: Any, size: Optional[int] = None
) -> int:
    return drm_command(fd, _IOC_READ | _IOC_WRITE, command_index, arg, size)


def drm_get_version(fd: Any, arg: drm_version) -> int:
    return drm_ioctl(fd, DRM_IOCTL_VERSION, arg)


def drm_gem_close(fd: Any, arg: drm_gem_close_req) -> int:
    return drm_ioctl(fd, DRM_IOCTL_GEM_CLOSE, arg)
