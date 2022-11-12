from mmap import MAP_ANONYMOUS, MAP_PRIVATE, MAP_SHARED, PROT_READ, PROT_WRITE
from typing import Union
from command_buffer import CommandBuffer, SubmittedCommandBufferBase
from drm import *
from gpu_memory import GpuMemoryBase
from nouveau_header import *
from utils import check_result
from ctypes import CDLL, c_void_p, c_size_t, c_int, c_int64, cast, memmove

libc = CDLL("libc.so.6")
mmap64 = libc.mmap64
# extern void *mmap64 (void *__addr, size_t __len, int __prot, int __flags, int __fd, __off64_t __offset) __THROW;
mmap64.argtypes = [c_void_p, c_size_t, c_int, c_int, c_int, c_int64]
mmap64.restype = c_int64

NOUVEAU_OBJECT_CURRENT_ID = 0xF000


def get_next_nouveau_object_id() -> int:
    global NOUVEAU_OBJECT_CURRENT_ID

    NOUVEAU_OBJECT_CURRENT_ID += 1

    return NOUVEAU_OBJECT_CURRENT_ID


class NouveauObject(object):
    parent: Optional["NouveauObject"]
    handle: int
    oclass: int
    fd: Any
    object: int

    def __init__(
        self,
        parent: Optional["NouveauObject"],
        handle: int,
        oclass: int,
        fd: Any,
        object: int,
    ) -> None:
        self.parent = parent
        self.handle = handle
        self.oclass = oclass
        self.fd = fd
        self.object = object

    @staticmethod
    def new(
        parent: "NouveauObject",
        handle: int,
        oclass: int,
        data: Any,
        constructor: Any = None,
        token: int = 0,
    ) -> Optional["NouveauObject"]:
        object_id = get_next_nouveau_object_id()

        ret = check_result(
            nvif_ioctl_v0_new(
                parent.fd, parent.object, token, object_id, handle, oclass, data
            ),
            0,
        )

        if ret == 0:
            if constructor is None:
                return NouveauObject(parent, handle, oclass, parent.fd, object_id)
            else:
                return constructor(parent, handle, oclass, parent.fd, object_id)

        return None

    def close(self):
        check_result(nvif_ioctl_v0_del(self.fd, self.object), None)

    def method(self, mthd: int, data: Any) -> int:
        return nvif_ioctl_v0_mthd(self.fd, self.object, mthd, data)

    def sclass(self, token: int, count) -> List[nvif_ioctl_sclass_oclass_v0]:
        output = list()

        ret = nvif_ioctl_v0_sclass(self.fd, self.object, token, count, output)

        if ret == 0:
            return output

        return None


class NvDeviceObject(NouveauObject):
    def __init__(
        self,
        parent: Optional["NouveauObject"],
        handle: int,
        oclass: int,
        fd: Any,
        object: int,
    ) -> None:
        super().__init__(parent, handle, oclass, fd, object)

    def device_info(self) -> Optional[nv_device_info_v0]:
        res = nv_device_info_v0()
        res.version = 0

        ret = self.method(NV_DEVICE_V0_INFO, res)

        if ret != 0:
            return None

        return res

    def time(self) -> Optional[nv_device_time_v0]:
        res = nv_device_time_v0()
        res.version = 0

        return check_result(self.method(NV_DEVICE_V0_TIME, res), res)


class NouveauSubchannel(NouveauObject):
    def __init__(
        self,
        parent: Optional["NouveauObject"],
        handle: int,
        oclass: int,
        fd: Any,
        object: int,
    ) -> None:
        super().__init__(parent, handle, oclass, fd, object)


class DrmNouveauObject(NouveauObject):
    def __init__(self, fd: Any) -> None:
        super().__init__(None, 0, 0, fd, 0)

    def create_device(self, device: int) -> Optional[NvDeviceObject]:
        req = nv_device_v0()
        req.version = 0
        req.device = device

        return NouveauObject.new(self, 0, NV_DEVICE, req, NvDeviceObject)


class NouveauBufferObject(object):
    fd: Any
    size: int
    offset: int
    handle: int
    map_handle: int
    domain: int

    def __init__(
        self, fd: Any, size: int, offset: int, handle: int, map_handle: int, domain: int
    ) -> None:
        self.fd = fd
        self.size = size
        self.offset = offset
        self.handle = handle
        self.map_handle = map_handle
        self.domain = domain

    def destroy(self):
        arg = drm_gem_close_req()
        arg.handle = self.handle

        check_result(drm_gem_close(self.fd, arg), None)

    def wait(self, flags: int) -> bool:
        req = drm_nouveau_gem_cpu_prep_req()
        req.handle = self.handle
        req.flags = flags

        res = drm_nouveau_gem_cpu_prep(self.fd, req)

        return res == 0

    def as_pushbuf(self, is_read: bool, is_write: bool) -> "NouveauPushbuf":
        domain = self.domain & 0x7

        read_domain = 0
        write_domain = 0

        if is_read:
            read_domain = domain

        if is_write:
            write_domain = domain

        return NouveauPushbuf(self, domain, read_domain, write_domain)


class NouveauBufferMappable(NouveauBufferObject, GpuMemoryBase):
    mapped_address: c_int64

    def __init__(
        self, fd: Any, size: int, offset: int, handle: int, map_handle: int, domain: int
    ) -> None:
        NouveauBufferObject.__init__(self, fd, size, offset, handle, map_handle, domain)

        res = mmap64(
            c_void_p(0), size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, map_handle
        )
        check_result(res, 0)

        self.mapped_address = res

        GpuMemoryBase.__init__(self, offset, size)

    def __getitem__(self, index: Union[slice, int]) -> Union[bytes, int]:
        temp_addr = cast(self.mapped_address, POINTER(c_ubyte))
        res = temp_addr.__getitem__(index)

        if type(res) is int:
            return res

        return bytes(bytearray(res))

    def __setitem__(self, index: Union[slice, int], object: Union[bytes, int]) -> None:
        if type(index) is int:
            start_offset = index
            end_offset = index + 1
        else:
            start_offset = index.start
            end_offset = index.stop

        if type(object) is int:
            object = bytearray([object])

        max_size = end_offset - start_offset

        temp_addr = cast(self.mapped_address + start_offset, POINTER(c_ubyte))
        raw_buff = (c_ubyte * max_size).from_buffer(object)

        memmove(temp_addr, raw_buff, max_size)

    def close(self):
        self.destroy()

    def __repr__(self) -> str:
        return f"NouveauBufferMappable(size=0x{self.size:x}, offset=0x{self.offset:x}, handle=0x{self.handle:x}, map_handle=0x{self.map_handle:x}, domain=0x{self.domain:x})"


class NouveauPushbuf(object):
    buffer_object: NouveauBufferObject
    valid_domain: int
    read_domain: int
    write_domain: int

    def __init__(
        self,
        buffer_object: NouveauBufferObject,
        valid_domain: int,
        read_domain: int,
        write_domain: int,
    ) -> None:
        self.buffer_object = buffer_object
        self.valid_domain = valid_domain
        self.read_domain = read_domain
        self.write_domain = write_domain


class NouveauPushbufCommandBuffer(NouveauPushbuf):
    offset: int
    size: int

    def __init__(
        self,
        buffer_object: NouveauBufferObject,
        valid_domain: int,
        read_domain: int,
        write_domain: int,
        offset: int,
        size: int,
    ) -> None:
        super().__init__(buffer_object, valid_domain, read_domain, write_domain)
        self.offset = offset
        self.size = size


class NouveauCard(DrmCard):
    def __init__(self, path: Optional[str] = "/dev/dri/card0", fd: int = -1) -> None:
        super().__init__(path, fd)

    def create_channel(
        self, fb_ctxdma_handle: int = 0, tt_ctxdma_handle: int = 0
    ) -> Optional["NouveauChannel"]:
        req = drm_nouveau_channel_alloc_req()
        req.fb_ctxdma_handle = fb_ctxdma_handle
        req.tt_ctxdma_handle = tt_ctxdma_handle

        ret = drm_nouveau_channel_alloc(self.fd, req)

        if ret == 0:
            return NouveauChannel(self, req.channel)

        return None

    def create_buffer_object(
        self,
        size: int,
        align: int,
        domain: int,
        pte_kind: int = 0,
        tile_mode: int = 0,
        constructor=NouveauBufferMappable,
    ) -> Optional[NouveauBufferObject]:
        if align == 0:
            align = 0x1000

        req = drm_nouveau_gem_new_req()
        req.info.domain = domain
        req.info.tile_flags = pte_kind << 8
        req.info.tile_mode = tile_mode
        req.info.size = size
        req.align = align

        ret = drm_nouveau_gem_new(self.fd, req)

        if ret == 0:
            return constructor(
                self.fd,
                req.info.size,
                req.info.offset,
                req.info.handle,
                req.info.map_handle,
                req.info.domain,
            )

        return None

    def create_buffer_mappable(
        self, size: int, align: int, domain: int, pte_kind: int = 0, tile_mode: int = 0
    ) -> Optional[NouveauBufferMappable]:
        return self.create_buffer_object(
            size,
            align,
            domain | NOUVEAU_GEM_DOMAIN_MAPPABLE,
            pte_kind,
            tile_mode,
            NouveauBufferMappable,
        )

    def submit_pushbuf(
        self, channel: "NouveauChannel", buffer_objects: List[NouveauPushbuf]
    ):
        req = drm_nouveau_gem_pushbuf_req()

        pushbuf_count = 0

        for buffer_object in buffer_objects:
            if type(buffer_object) is NouveauPushbufCommandBuffer:
                pushbuf_count += 1

        buffers = (drm_nouveau_gem_pushbuf_bo * len(buffer_objects))()
        push = (drm_nouveau_gem_pushbuf_push * pushbuf_count)()

        push_index = 0

        for buffer_index in range(len(buffer_objects)):
            object = buffer_objects[buffer_index]

            buffers[buffer_index] = drm_nouveau_gem_pushbuf_bo()
            buffers[buffer_index].handle = object.buffer_object.handle
            buffers[buffer_index].read_domains = object.read_domain
            buffers[buffer_index].write_domains = object.write_domain
            buffers[buffer_index].valid_domains = object.valid_domain

            if type(buffer_object) is NouveauPushbufCommandBuffer:
                push[push_index] = drm_nouveau_gem_pushbuf_push()
                push[push_index].bo_index = buffer_index
                push[push_index].offset = buffer_object.offset
                push[push_index].length = buffer_object.size
                push_index += 1

        req.channel = channel.channel_id

        req.nr_buffers = len(buffer_objects)
        req.buffers = buffers

        req.nr_push = pushbuf_count
        req.push = push

        check_result(drm_nouveau_gem_pushbuf(self.fd, req), 0)

    def as_drm_nouveau_object(self) -> DrmNouveauObject:
        return DrmNouveauObject(self.fd)


class NouveauChannel(object):
    card: NouveauCard
    channel_id: int

    engine_copy: Optional[NouveauSubchannel]
    engine_2d: Optional[NouveauSubchannel]
    engine_3d: Optional[NouveauSubchannel]
    engine_m2mf: Optional[NouveauSubchannel]
    engine_compute: Optional[NouveauSubchannel]

    def __init__(self, card: NouveauCard, channel_id: int) -> None:
        self.card = card
        self.channel_id = channel_id
        self.engine_copy = None
        self.engine_2d = None
        self.engine_3d = None
        self.engine_m2mf = None
        self.engine_compute = None

        drm_object = self.card.as_drm_nouveau_object()
        engine_infos = drm_object.sclass(self.channel_id, 10)

        if engine_infos is not None:
            for engine_info in engine_infos:
                engine = NouveauObject.new(
                    drm_object,
                    engine_info.oclass,
                    engine_info.oclass,
                    None,
                    NouveauSubchannel,
                    token=self.channel_id,
                )

                channel_class_category = engine_info.oclass & 0xFF

                if channel_class_category == 0xB5:
                    self.engine_copy = engine
                elif channel_class_category == 0x2D:
                    self.engine_2d = engine
                elif channel_class_category == 0x97:
                    self.engine_3d = engine
                elif channel_class_category in [0x39, 0x40]:
                    self.engine_m2mf = engine
                elif channel_class_category == 0xC0:
                    self.engine_compute = engine
                else:
                    print(f"Unknown engine {engine_info.oclass:x}")
                    engine.close()

            assert self.engine_copy
            assert self.engine_2d
            assert self.engine_3d
            assert self.engine_m2mf
            assert self.engine_compute

    def close(self):
        if self.engine_copy is not None:
            self.engine_copy.close()

        if self.engine_2d is not None:
            self.engine_2d.close()

        if self.engine_3d is not None:
            self.engine_3d.close()

        if self.engine_m2mf is not None:
            self.engine_m2mf.close()

        if self.engine_compute is not None:
            self.engine_compute.close()

        req = drm_nouveau_channel_free_req()
        req.channel = self.channel_id

        check_result(drm_nouveau_channel_free(self.card.fd, req), None)


class NouveauSubmittedCommandBuffer(SubmittedCommandBufferBase):
    dummy_memory: NouveauBufferMappable

    def __init__(
        self,
        command_buffer_memory: NouveauBufferMappable,
        dummy_memory: NouveauBufferMappable,
    ) -> None:
        super().__init__(command_buffer_memory)

        self.dummy_memory = dummy_memory

    def wait(self, timeout: Optional[float] = None) -> bool:
        return self.dummy_memory.wait(NOUVEAU_GEM_CPU_PREP_WRITE)

    def close(self):
        self.dummy_memory.close()
        self.gpu_memory.close()


class NouveauGpuChannel(object):
    PAGE_SIZE: int = 0x1000

    card: NouveauCard
    drm_nouveau_object: DrmNouveauObject
    device_object: NvDeviceObject
    device_info: nv_device_info_v0
    internal_channel: NouveauChannel

    def __init__(self, path: Optional[str] = "/dev/dri/card0", fd: int = -1) -> None:
        self.card = NouveauCard(path, fd)
        self.drm_nouveau_object = self.card.as_drm_nouveau_object()

        self.device_object = self.drm_nouveau_object.create_device(~0)
        assert self.device_object is not None

        self.device_info = self.device_object.device_info()
        assert self.device_info is not None

        self.internal_channel = self.card.create_channel()
        assert self.internal_channel is not None

    def close(self):
        self.internal_channel.close()
        self.device_object.close()
        self.card.close()

    def create_gpu_memory(self, size: int) -> NouveauBufferMappable:
        res = self.card.create_buffer_mappable(size, 0, NOUVEAU_GEM_DOMAIN_GART)

        if res is None:
            raise Exception("cannot create GPU memory!")

        return res

    def submit_commands(
        self,
        command_buffers: List[CommandBuffer],
        external_wait: Optional[SubmittedCommandBufferBase] = None,
        buffer_deps: Optional[List[NouveauPushbuf]] = None,
    ) -> List[NouveauSubmittedCommandBuffer]:
        # TODO: wait on GPU
        if external_wait is not None:
            assert external_wait.wait()

        buffers = list()
        command_buffer_memory_list = list()
        dummy_buffer_memory_list = list()

        for command_buffer in command_buffers:
            memory = self.create_gpu_memory(len(command_buffer.buffer))
            memory[0 : len(command_buffer.buffer)] = command_buffer.buffer

            command_buffer_memory_list.append(memory)
            buffers.append(NouveauPushbuf(memory, memory.domain, memory.domain, 0))

            dummy_memory = self.create_gpu_memory(1)
            dummy_buffer_memory_list.append(dummy_memory)
            buffers.append(dummy_memory.as_pushbuf(True, True))

        if buffer_deps is not None:
            buffers.extend(buffer_deps)

        self.card.submit_pushbuf(self.internal_channel, buffers)

        res = list()

        for command_buffer_index in range(len(command_buffers)):
            res.append(
                NouveauSubmittedCommandBuffer(
                    command_buffer_memory_list[command_buffer_index],
                    dummy_buffer_memory_list[command_buffer_index],
                )
            )

        return res

    def submit_command(
        self,
        command_buffer: CommandBuffer,
        external_wait: Optional[SubmittedCommandBufferBase] = None,
    ) -> NouveauSubmittedCommandBuffer:
        return self.submit_commands([command_buffer], external_wait)[0]
