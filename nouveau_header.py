from ctypes import (
    Structure,
    c_ubyte,
    c_uint8,
    c_uint16,
    c_uint32,
    c_uint64,
    c_int,
    c_int16,
    c_int32,
    POINTER,
    sizeof,
)
from typing import Any, List, Optional
from drm_header import drm_command_read_write
from utils import bytearray_to_ctypes_struct, ctypes_struct_to_bytearray

DRM_NOUVEAU_GETPARAM = 0x00
DRM_NOUVEAU_SETPARAM = 0x01
DRM_NOUVEAU_CHANNEL_ALLOC = 0x02
DRM_NOUVEAU_CHANNEL_FREE = 0x03
DRM_NOUVEAU_GROBJ_ALLOC = 0x04
DRM_NOUVEAU_NOTIFIEROBJ_ALLOC = 0x05
DRM_NOUVEAU_GPUOBJ_FREE = 0x06
DRM_NOUVEAU_NVIF = 0x07
DRM_NOUVEAU_SVM_INIT = 0x08
DRM_NOUVEAU_SVM_BIND = 0x09
DRM_NOUVEAU_GEM_NEW = 0x40
DRM_NOUVEAU_GEM_PUSHBUF = 0x41
DRM_NOUVEAU_GEM_CPU_PREP = 0x42
DRM_NOUVEAU_GEM_CPU_FINI = 0x43
DRM_NOUVEAU_GEM_INFO = 0x44


NOUVEAU_GEM_DOMAIN_CPU = 1 << 0
NOUVEAU_GEM_DOMAIN_VRAM = 1 << 1
NOUVEAU_GEM_DOMAIN_GART = 1 << 2
NOUVEAU_GEM_DOMAIN_MAPPABLE = 1 << 3
NOUVEAU_GEM_DOMAIN_COHERENT = 1 << 4
NOUVEAU_GEM_TILE_COMP = 0x00030000
NOUVEAU_GEM_TILE_LAYOUT_MASK = 0x0000FF00
NOUVEAU_GEM_TILE_16BPP = 0x00000001
NOUVEAU_GEM_TILE_32BPP = 0x00000002
NOUVEAU_GEM_TILE_ZETA = 0x00000004
NOUVEAU_GEM_TILE_NONCONTIG = 0x00000008

NOUVEAU_GEM_RELOC_LOW = 1 << 0
NOUVEAU_GEM_RELOC_HIGH = 1 << 1
NOUVEAU_GEM_RELOC_OR = 1 << 2

NOUVEAU_GEM_CPU_PREP_NOWAIT = 0x1
NOUVEAU_GEM_CPU_PREP_WRITE = 0x4


class drm_nouveau_channel_alloc_subchannel(Structure):
    _fields_ = [("handle", c_uint32), ("grclass", c_uint32)]


class drm_nouveau_channel_alloc_req(Structure):
    _fields_ = [
        ("fb_ctxdma_handle", c_uint32),
        ("tt_ctxdma_handle", c_uint32),
        ("channel", c_int),
        ("pushbuf_domains", c_uint32),
        ("notifier_handle", c_uint32),
        ("subchan", drm_nouveau_channel_alloc_subchannel * 8),
        ("nr_subchan", c_uint32),
    ]


class drm_nouveau_channel_free_req(Structure):
    _fields_ = [
        ("channel", c_int),
    ]


def drm_nouveau_channel_alloc(fd: Any, arg: drm_nouveau_channel_alloc_req) -> int:
    return drm_command_read_write(fd, DRM_NOUVEAU_CHANNEL_ALLOC, arg)


def drm_nouveau_channel_free(fd: Any, arg: drm_nouveau_channel_free_req) -> int:
    return drm_command_read_write(fd, DRM_NOUVEAU_CHANNEL_FREE, arg)


NVIF_IOCTL_V0_NOP = 0x00
NVIF_IOCTL_V0_SCLASS = 0x01
NVIF_IOCTL_V0_NEW = 0x02
NVIF_IOCTL_V0_DEL = 0x03
NVIF_IOCTL_V0_MTHD = 0x04
NVIF_IOCTL_V0_RD = 0x05
NVIF_IOCTL_V0_WR = 0x06
NVIF_IOCTL_V0_MAP = 0x07
NVIF_IOCTL_V0_UNMAP = 0x08
NVIF_IOCTL_V0_NTFY_NEW = 0x09
NVIF_IOCTL_V0_NTFY_DEL = 0x0A
NVIF_IOCTL_V0_NTFY_GET = 0x0B
NVIF_IOCTL_V0_NTFY_PUT = 0x0C


NVIF_IOCTL_V0_OWNER_NVIF = 0x00
NVIF_IOCTL_V0_OWNER_ANY = 0xFF


NVIF_IOCTL_V0_ROUTE_NVIF = 0x00
NVIF_IOCTL_V0_ROUTE_HIDDEN = 0xFF


NV_DEVICE = 0x00000080


class nvif_ioctl_v0(Structure):
    _fields_ = [
        ("version", c_uint8),
        ("type", c_uint8),
        ("pad02", c_uint8 * 4),
        ("owner", c_uint8),
        ("route", c_uint8),
        ("token", c_uint64),
        ("object", c_uint64),
        # NOTE: Data start here
    ]


def __create_nvif_ioctl_struct(
    req_type: int, object: int = 0, token: int = 0
) -> nvif_ioctl_v0:
    request = nvif_ioctl_v0()
    request.version = 0
    request.type = req_type

    if token == 0:
        request.owner = NVIF_IOCTL_V0_OWNER_ANY
        request.route = NVIF_IOCTL_V0_ROUTE_NVIF
        request.object = object
    else:
        request.owner = NVIF_IOCTL_V0_OWNER_NVIF
        request.route = NVIF_IOCTL_V0_OWNER_ANY
        request.token = token

    return request


class nvif_ioctl_nop_v0(Structure):
    _fields_ = [("version", c_uint8)]


class nvif_ioctl_sclass_oclass_v0(Structure):
    _fields_ = [
        ("oclass", c_int32),
        ("minver", c_int16),
        ("maxver", c_int16),
    ]


class nvif_ioctl_sclass_v0(Structure):
    _fields_ = [
        ("version", c_uint8),
        ("count", c_uint8),
        ("pad02", c_uint8 * 6),
        # NOTE: result data start here
    ]


class nvif_ioctl_new_v0(Structure):
    _fields_ = [
        ("version", c_uint8),
        ("pad01", c_uint8 * 6),
        ("route", c_uint8),
        ("token", c_uint64),
        ("object", c_uint64),
        ("handle", c_uint32),
        ("oclass", c_int32),
        # NOTE: class data start here
    ]


class nvif_ioctl_del(Structure):
    _fields_ = []


class nvif_ioctl_mthd_v0(Structure):
    _fields_ = [
        ("version", c_uint8),
        ("method", c_uint8),
        ("pad02", c_uint8 * 6),
        # NOTE: data start here
    ]


def drm_nouveau_nvif(fd: Any, arg: Any, size: Optional[int] = None) -> int:
    return drm_command_read_write(fd, DRM_NOUVEAU_NVIF, arg, size)


class drm_nouveau_gem_info(Structure):
    _fields_ = [
        ("handle", c_uint32),
        ("domain", c_uint32),
        ("size", c_uint64),
        ("offset", c_uint64),
        ("map_handle", c_uint64),
        ("tile_mode", c_uint32),
        ("tile_flags", c_uint32),
    ]


class drm_nouveau_gem_new_req(Structure):
    _fields_ = [
        ("info", drm_nouveau_gem_info),
        ("channel_hint", c_uint32),
        ("align", c_uint32),
    ]


def drm_nouveau_gem_new(fd: Any, req: drm_nouveau_gem_new_req) -> int:
    return drm_command_read_write(fd, DRM_NOUVEAU_GEM_NEW, req)


class drm_nouveau_gem_pushbuf_bo_presumed(Structure):
    _fields_ = [
        ("valid", c_uint32),
        ("domain", c_uint32),
        ("offset", c_uint32),
    ]


class drm_nouveau_gem_pushbuf_bo(Structure):
    _fields_ = [
        ("user_priv", c_uint64),
        ("handle", c_uint32),
        ("read_domains", c_uint32),
        ("write_domains", c_uint32),
        ("valid_domains", c_uint32),
        ("presumed", drm_nouveau_gem_pushbuf_bo_presumed),
    ]


class drm_nouveau_gem_pushbuf_reloc(Structure):
    _fields_ = [
        ("reloc_bo_index", c_uint32),
        ("reloc_bo_offset", c_uint32),
        ("bo_index", c_uint32),
        ("flags", c_uint32),
        ("data", c_uint32),
        ("vor", c_uint32),
        ("tor", c_uint32),
    ]


class drm_nouveau_gem_pushbuf_push(Structure):
    _fields_ = [
        ("bo_index", c_uint32),
        ("pad", c_uint32),
        ("offset", c_uint64),
        ("length", c_uint64),
    ]


class drm_nouveau_gem_pushbuf_req(Structure):
    _fields_ = [
        ("channel", c_uint32),
        ("nr_buffers", c_uint32),
        ("buffers", POINTER(drm_nouveau_gem_pushbuf_bo)),
        ("nr_relocs", c_uint32),
        ("nr_push", c_uint32),
        ("relocs", c_uint64),
        ("push", POINTER(drm_nouveau_gem_pushbuf_push)),
        ("suffix0", c_uint32),
        ("suffix1", c_uint32),
        ("vram_available", c_uint64),
        ("gart_available", c_uint64),
    ]


def drm_nouveau_gem_pushbuf(fd: Any, req: drm_nouveau_gem_pushbuf_req) -> int:
    return drm_command_read_write(fd, DRM_NOUVEAU_GEM_PUSHBUF, req)


class drm_nouveau_gem_cpu_prep_req(Structure):
    _fields_ = [
        ("handle", c_uint32),
        ("flags", c_uint32),
    ]


def drm_nouveau_gem_cpu_prep(fd: Any, req: drm_nouveau_gem_cpu_prep_req) -> int:
    return drm_command_read_write(fd, DRM_NOUVEAU_GEM_CPU_PREP, req)


class drm_nouveau_gem_cpu_fini_req(Structure):
    _fields_ = [
        ("handle", c_uint32),
    ]


def drm_nouveau_gem_cpu_fini(fd: Any, req: drm_nouveau_gem_cpu_fini_req) -> int:
    return drm_command_read_write(fd, DRM_NOUVEAU_GEM_CPU_FINI, req)


class nvif_ioctl_v0_new_req(Structure):
    _fields_ = [("ioctl", nvif_ioctl_v0), ("new", nvif_ioctl_new_v0)]


class nvif_ioctl_v0_sclass_req(Structure):
    _fields_ = [("ioctl", nvif_ioctl_v0), ("sclass", nvif_ioctl_sclass_v0)]


def nvif_ioctl_v0_sclass(
    fd: Any,
    object: int,
    token: int,
    count: int,
    output_array: List[nvif_ioctl_sclass_oclass_v0],
) -> int:
    request = nvif_ioctl_v0_sclass_req()
    request.ioctl = __create_nvif_ioctl_struct(
        NVIF_IOCTL_V0_SCLASS, object=object, token=token
    )
    request.sclass.version = 0
    request.sclass.count = count

    raw_arg = ctypes_struct_to_bytearray(request)
    raw_arg += bytearray(sizeof(nvif_ioctl_sclass_oclass_v0) * count)
    buff = (c_ubyte * len(raw_arg)).from_buffer(raw_arg)

    ret = drm_nouveau_nvif(fd, buff, len(buff))

    extra_data = raw_arg[sizeof(request) :]

    if ret == 0:
        bytearray_to_ctypes_struct(request, raw_arg[: sizeof(request)])

        for _ in range(request.sclass.count):
            entry = nvif_ioctl_sclass_oclass_v0()
            bytearray_to_ctypes_struct(entry, extra_data)
            output_array.append(entry)
            extra_data = extra_data[sizeof(nvif_ioctl_sclass_oclass_v0) :]

    return ret


def nvif_ioctl_v0_new(
    fd: Any,
    parent_object: int,
    token: int,
    object: int,
    handle: int,
    oclass: int,
    data: Any,
) -> int:
    request = nvif_ioctl_v0_new_req()
    request.ioctl = __create_nvif_ioctl_struct(
        NVIF_IOCTL_V0_NEW, object=parent_object, token=token
    )
    request.new.version = 0
    request.new.route = NVIF_IOCTL_V0_ROUTE_NVIF
    request.new.token = token
    request.new.object = object
    request.new.handle = handle
    request.new.oclass = oclass

    raw_arg = ctypes_struct_to_bytearray(request)

    if data is not None:
        raw_arg += ctypes_struct_to_bytearray(data)

    buff = (c_ubyte * len(raw_arg)).from_buffer(raw_arg)

    ret = drm_nouveau_nvif(fd, buff, len(buff))

    if data is not None:
        bytearray_to_ctypes_struct(data, raw_arg[sizeof(request) :])

    return ret


class nvif_ioctl_del_req(Structure):
    _fields_ = [("ioctl", nvif_ioctl_v0), ("del", nvif_ioctl_del)]


def nvif_ioctl_v0_del(fd: Any, object: int) -> int:
    request = nvif_ioctl_del_req()
    request.ioctl = __create_nvif_ioctl_struct(NVIF_IOCTL_V0_DEL, object)

    return drm_nouveau_nvif(fd, request, sizeof(nvif_ioctl_v0))


class nvif_ioctl_mthd_req(Structure):
    _fields_ = [("ioctl", nvif_ioctl_v0), ("mthd", nvif_ioctl_mthd_v0)]


def nvif_ioctl_v0_mthd(fd: Any, object: int, mthd: int, data: Any) -> int:
    request = nvif_ioctl_mthd_req()
    request.ioctl = __create_nvif_ioctl_struct(NVIF_IOCTL_V0_MTHD, object)
    request.mthd.version = 0
    request.mthd.method = mthd

    raw_arg = ctypes_struct_to_bytearray(request)
    raw_arg += ctypes_struct_to_bytearray(data)
    buff = (c_ubyte * len(raw_arg)).from_buffer(raw_arg)

    ret = drm_nouveau_nvif(fd, buff, len(buff))

    bytearray_to_ctypes_struct(data, raw_arg[sizeof(request) :])

    return ret


class nv_device_v0(Structure):
    _fields_ = [("version", c_uint8), ("pad01", c_uint8 * 7), ("device", c_uint64)]


NV_DEVICE_V0_INFO = 0x00
NV_DEVICE_V0_TIME = 0x01


class nv_device_info_v0(Structure):
    _fields_ = [
        ("version", c_uint8),
        ("platform", c_uint8),
        ("chipset", c_uint16),
        ("revision", c_uint8),
        ("family", c_uint8),
        ("pad06", c_uint8 * 2),
        ("ram_size", c_uint64),
        ("ram_user", c_uint64),
        ("chip", c_ubyte * 16),
        ("name", c_ubyte * 64),
    ]


class nv_device_time_v0(Structure):
    _fields_ = [("version", c_uint8), ("pad01", c_uint8 * 7), ("time", c_uint64)]
