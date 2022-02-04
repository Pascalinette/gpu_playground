from ctypes import (
    addressof,
    c_short,
    c_uint,
    c_int,
    c_ulong,
    sizeof,
    POINTER,
    CDLL,
)
from mmap import MAP_SHARED, PROT_READ, PROT_WRITE, mmap
from select import *
from typing import Any, List, Optional, Tuple, Union, overload
from command_buffer import *
from nvmap_header import (
    NVMAP_IOC_FREE,
    nvmap_alloc_handle,
    nvmap_create_handle,
    nvmap_ioc_alloc,
    nvmap_ioc_create,
    nvmap_ioc_get_fd,
)
from nvgpu_header import (
    NVGPU_AS_MAP_BUFFER_FLAGS_CACHEABLE,
    NVGPU_AS_MAP_BUFFER_FLAGS_DIRECT_KIND_CTRL,
    NVGPU_SUBMIT_GPFIFO_FLAGS_FENCE_GET,
    NVGPU_SUBMIT_GPFIFO_FLAGS_FENCE_WAIT,
    NVGPU_SUBMIT_GPFIFO_FLAGS_SYNC_FENCE,
    nvgpu_alloc_as_args,
    nvgpu_alloc_gpfifo_args,
    nvgpu_alloc_obj_ctx_args,
    nvgpu_as_bind_channel_args,
    nvgpu_as_ioctl_bind_channel,
    nvgpu_as_ioctl_map_buffer_ex,
    nvgpu_as_map_buffer_ex_args,
    nvgpu_channel_setup_bind_args,
    nvgpu_fence,
    nvgpu_gpu_characteristics,
    nvgpu_gpu_get_characteristics,
    nvgpu_gpu_ioctl_alloc_as,
    nvgpu_gpu_ioctl_get_characteristics,
    nvgpu_gpu_ioctl_open_channel,
    nvgpu_gpu_ioctl_open_tsg,
    nvgpu_gpu_open_channel_args,
    nvgpu_gpu_open_tsg_args,
    nvgpu_ioctl_channel_alloc_gpfifo,
    nvgpu_ioctl_channel_alloc_obj_ctx,
    nvgpu_ioctl_channel_set_nvmap_fd,
    nvgpu_ioctl_channel_setup_bind,
    nvgpu_ioctl_channel_submit_gpfifo,
    nvgpu_set_nvmap_fd_args,
    nvgpu_submit_gpfifo_args,
    nvgpu_tsg_ioctl_bind_channel,
)
from os import close
import errno

libc = CDLL("libc.so.6")
get_errno_loc = libc.__errno_location
get_errno_loc.restype = POINTER(c_int)

ioctl = libc.ioctl


def align_up(value: int, size: int) -> int:
    return (value + (size - 1)) & -size


def align_down(value: int, size: int) -> int:
    return value & -size


class ErrnoException(Exception):
    """Raised when an errno is returned"""

    error: int
    message: Optional[str]

    def __init__(self, error: int) -> None:
        self.error = error

        if error in errno.errorcode:
            self.message = errno.errorcode[error]

        super().__init__(error)


def get_errno() -> int:
    return get_errno_loc()[0]


def get_errno_code() -> Optional[str]:
    error = get_errno()

    if error != 0:
        return errno.errorcode[error]

    return None


class BlockDevice(object):
    file: Any
    fd: int

    def __init__(self, path: Optional[str] = None, fd: int = -1) -> None:
        if fd != -1:
            self.fd = fd
        elif path is not None:
            self.file = open(path, "wb")
            self.fd = self.file.fileno()
        else:
            raise Exception("INVALID COMBINAISON")

    def close(self) -> None:
        if self.file is not None:
            self.file.close()
        else:
            close(self.fd)

    def check_result(self, result_code: int, output_value: Any = None) -> Any:
        if result_code == 0:
            return output_value

        exception = ErrnoException(get_errno())

        print(exception.message)

        raise exception


class NvMap(BlockDevice):
    def __init__(self) -> None:
        super().__init__("/dev/nvmap")

    def create(self, size: int) -> int:
        request = nvmap_create_handle()
        request.unamed_field0.unamed_field0.unamed_field0.size = size

        self.check_result(nvmap_ioc_create(self.fd, request))

        return request.unamed_field0.unamed_field0.handle

    def get_fd(self, handle: int) -> int:
        request = nvmap_create_handle()
        request.unamed_field0.unamed_field0.handle = handle

        self.check_result(nvmap_ioc_get_fd(self.fd, request))

        return request.unamed_field0.unamed_field0.unamed_field0.fd

    def free(self, handle: int) -> None:
        self.check_result(ioctl(self.fd, NVMAP_IOC_FREE, c_uint(handle)))

    def allocate(self, handle: int, heap_mask: int, flags: int, align: int) -> None:
        request = nvmap_alloc_handle()
        request.handle = handle
        request.heap_mask = heap_mask
        request.flags = flags
        request.align = align

        self.check_result(nvmap_ioc_alloc(self.fd, request))


class NvHostGpu(BlockDevice):
    def __init__(self, path: Optional[str] = "/dev/nvhost-gpu", fd: int = -1) -> None:
        super().__init__(path, fd)

    def set_nvmap_fd(self, nvmap: NvMap) -> None:
        request = nvgpu_set_nvmap_fd_args()
        request.fd = nvmap.fd

        self.check_result(nvgpu_ioctl_channel_set_nvmap_fd(self.fd, request))

    def alloc_gpfifo(self, num_entries: int, flags: int) -> None:
        request = nvgpu_alloc_gpfifo_args()
        request.num_entries = num_entries
        request.flags = flags

        self.check_result(nvgpu_ioctl_channel_alloc_gpfifo(self.fd, request))

    def setup_bind(
        self,
        num_gpfifo_entries: int,
        num_inflight_jobs: int,
        flags: int,
        userd_dmabuf_fd: int = 0,
        gpfifo_dmabuf_fd: int = 0,
        userd_dmabuf_offset: int = 0,
        gpfifo_dmabuf_offset: int = 0,
    ) -> int:
        request = nvgpu_channel_setup_bind_args()
        request.num_gpfifo_entries = num_gpfifo_entries
        request.num_inflight_jobs = num_inflight_jobs
        request.flags = flags
        request.userd_dmabuf_fd = userd_dmabuf_fd
        request.gpfifo_dmabuf_fd = gpfifo_dmabuf_fd
        request.userd_dmabuf_offset = userd_dmabuf_offset
        request.gpfifo_dmabuf_offset = gpfifo_dmabuf_offset

        self.check_result(nvgpu_ioctl_channel_setup_bind(self.fd, request))

        if errno == 0:
            return request.work_submit_token

    def alloc_obj_ctx(self, class_num: int, flags: int) -> int:
        request = nvgpu_alloc_obj_ctx_args()
        request.class_num = class_num
        request.flags = flags

        self.check_result(nvgpu_ioctl_channel_alloc_obj_ctx(self.fd, request))

        return request.obj_id

    def submit_gpfifo(
        self,
        user_queue: List[c_ulong],
        flags: int,
        waiting_fence: Optional[nvgpu_fence] = None,
    ) -> Optional[nvgpu_fence]:
        queue_array_type = c_ulong * len(user_queue)
        ioctl_user_queue = queue_array_type(*user_queue)

        request = nvgpu_submit_gpfifo_args()
        request.gpfifo = addressof(ioctl_user_queue)
        request.num_entries = len(user_queue)
        request.flags = flags

        if (
            waiting_fence is not None
            and (flags & NVGPU_SUBMIT_GPFIFO_FLAGS_FENCE_WAIT)
            == NVGPU_SUBMIT_GPFIFO_FLAGS_FENCE_WAIT
        ):
            request.fence = waiting_fence

        self.check_result(nvgpu_ioctl_channel_submit_gpfifo(self.fd, request))

        if (
            flags & NVGPU_SUBMIT_GPFIFO_FLAGS_FENCE_GET
        ) == NVGPU_SUBMIT_GPFIFO_FLAGS_FENCE_GET:
            return request.fence

        return None


class NvAddressSpace(BlockDevice):
    def __init__(
        self, path: Optional[str] = "/dev/nvhost-as-gpu", fd: int = -1
    ) -> None:
        super().__init__(path, fd)

    def bind_channel(self, channel: NvHostGpu) -> int:
        request = nvgpu_as_bind_channel_args()
        request.channel_fd = channel.fd

        errno = nvgpu_as_ioctl_bind_channel(self.fd, request)

        if errno == 0:
            return 0

        return get_errno()

    def map_buffer_ex(
        self,
        dmabuf_fd: int,
        flags: int,
        page_size: int = 0,
        compr_kind: int = -1,
        incompr_kind: int = -1,
        buffer_offset: int = 0,
        mapping_size: int = 0,
        offset: int = 0,
    ) -> Tuple[int, int, int]:
        request = nvgpu_as_map_buffer_ex_args()
        request.flags = flags
        request.compr_kind = c_short(compr_kind)
        request.incompr_kind = c_short(incompr_kind)
        request.dmabuf_fd = dmabuf_fd
        request.page_size = page_size
        request.buffer_offset = buffer_offset
        request.mapping_size = mapping_size
        request.offset = offset

        self.check_result(nvgpu_as_ioctl_map_buffer_ex(self.fd, request))

        return (request.flags, request.page_size, request.offset)


class NvHostTSGGpu(BlockDevice):
    def __init__(
        self, path: Optional[str] = "/dev/nvhost-tsg-gpu", fd: int = -1
    ) -> None:
        super().__init__(path, fd)

    def bind_channel(self, channel: NvHostGpu) -> None:
        self.check_result(nvgpu_tsg_ioctl_bind_channel(self.fd, c_int(channel.fd)))


class NvHostGpuCtrl(BlockDevice):
    def __init__(self) -> None:
        super().__init__("/dev/nvhost-ctrl-gpu")

    def get_characteristics(self) -> Union[nvgpu_gpu_characteristics, int]:
        result = nvgpu_gpu_characteristics()

        request = nvgpu_gpu_get_characteristics()
        request.gpu_characteristics_buf_addr = addressof(result)
        request.gpu_characteristics_buf_size = sizeof(result)

        return self.check_result(
            nvgpu_gpu_ioctl_get_characteristics(self.fd, request), result
        )

    def allocate_address_space(
        self, big_page_size: c_uint, flags: c_uint
    ) -> NvAddressSpace:
        request = nvgpu_alloc_as_args()
        request.big_page_size = big_page_size
        request.flags = flags

        self.check_result(nvgpu_gpu_ioctl_alloc_as(self.fd, request))

        return NvAddressSpace(request.as_fd)

    def open_tsg(self) -> NvHostTSGGpu:
        request = nvgpu_gpu_open_tsg_args()

        self.check_result(nvgpu_gpu_ioctl_open_tsg(self.fd, request))

        return NvHostTSGGpu(request.tsg_fd)

    def open_channel(self, runlist_id: int) -> Union[NvHostGpu, int]:
        request = nvgpu_gpu_open_channel_args()
        request.unamed_field0.runlist_id = runlist_id

        self.check_result(nvgpu_gpu_ioctl_open_channel(self.fd, request))

        return NvHostGpu(request.unamed_field0.channel_fd)


class GpuMemory(object):
    nvmap_instance: NvMap
    nvmap_handle: int
    user_size: int
    mmap_instance: mmap

    address_space: NvAddressSpace
    gpu_address: int
    gpu_memory_size: int

    def __init__(
        self,
        mmap_instance: mmap,
        nvmap_instance: NvMap,
        nvmap_handle: int,
        user_size: int,
        address_space: NvAddressSpace,
        gpu_address: int,
        gpu_memory_size: int,
    ) -> None:
        self.nvmap_instance = nvmap_instance
        self.nvmap_handle = nvmap_handle
        self.user_size = user_size
        self.mmap_instance = mmap_instance
        self.address_space = address_space
        self.gpu_address = gpu_address
        self.gpu_memory_size = gpu_memory_size

    def __getitem__(self, index: slice) -> bytes:
        return self.mmap_instance.__getitem__(index)

    def __setitem__(self, index: slice, object: bytes) -> None:
        return self.mmap_instance.__setitem__(index, object)

    def close(self):
        self.mmap_instance.close()
        # FIXME: missing unmap of the address space here
        self.nvmap_instance.free(self.nvmap_handle)


class SubmittedCommandBuffer(object):
    gpu_memory: GpuMemory
    external_wait_fd: int
    external_wait: poll

    def __init__(
        self, gpu_memory: GpuMemory, external_wait_fd: int, external_wait: poll
    ) -> None:
        self.gpu_memory = gpu_memory
        self.external_wait_fd = external_wait_fd
        self.external_wait = external_wait

    def wait(self, timeout: Optional[float] = None) -> bool:
        return len(self.external_wait.poll(timeout)) != 0

    def close(self):
        self.wait()
        self.gpu_memory.close()

        self.external_wait.unregister(self.external_wait_fd)
        close(self.external_wait_fd)


class TegraGpuChannel(object):
    PAGE_SIZE: int = 0x1000

    nvhost_gpu_ctrl: NvHostGpuCtrl
    nvmap: NvMap
    characteristics: nvgpu_gpu_characteristics
    address_space: NvAddressSpace
    thread_scheduler_group: NvHostTSGGpu
    channel: NvHostGpu

    worktoken: Optional[int]
    object_id: int

    def __init__(self, gpfifo_queue_size: int = 0x800) -> None:
        self.nvhost_gpu_ctrl = NvHostGpuCtrl()
        self.nvmap = NvMap()

        self.characteristics = self.nvhost_gpu_ctrl.get_characteristics()
        self.address_space = self.nvhost_gpu_ctrl.allocate_address_space(
            self.characteristics.big_page_size, 0
        )
        self.thread_scheduler_group = self.nvhost_gpu_ctrl.open_tsg()
        self.channel = self.nvhost_gpu_ctrl.open_channel(-1)

        self.channel.set_nvmap_fd(self.nvmap)
        self.address_space.bind_channel(self.channel)
        self.thread_scheduler_group.bind_channel(self.channel)

        try:
            self.worktoken = self.channel.setup_bind(gpfifo_queue_size * 4, 0, 0)
        except ErrnoException:
            self.channel.alloc_gpfifo(gpfifo_queue_size, 0)

        self.object_id = self.channel.alloc_obj_ctx(
            self.characteristics.threed_class, 0
        )

        # Finaly bind all channels
        setup_engines_command_buffer = CommandBuffer()
        setup_engines_command_buffer.write_u32(BIND_CHANNEL_3D)
        setup_engines_command_buffer.write_u32(self.characteristics.threed_class)
        setup_engines_command_buffer.write_u32(BIND_CHANNEL_COMPUTE)
        setup_engines_command_buffer.write_u32(self.characteristics.compute_class)
        setup_engines_command_buffer.write_u32(BIND_CHANNEL_I2M)
        setup_engines_command_buffer.write_u32(
            self.characteristics.inline_to_memory_class
        )
        setup_engines_command_buffer.write_u32(BIND_CHANNEL_2D)
        setup_engines_command_buffer.write_u32(self.characteristics.twod_class)
        setup_engines_command_buffer.write_u32(BIND_CHANNEL_DMA)
        setup_engines_command_buffer.write_u32(self.characteristics.dma_copy_class)
        setup_engines_submitted_command = self.submit_command(
            setup_engines_command_buffer
        )
        setup_engines_submitted_command.wait()
        setup_engines_submitted_command.close()

    def len(self) -> int:
        return self.user_size

    def close(self):
        self.channel.close()
        self.thread_scheduler_group.close()
        self.address_space.close()
        self.nvhost_gpu_ctrl.close()
        self.nvmap.close()

    def create_gpu_memory(
        self,
        size: int,
        is_cpu_cached: bool = True,
        is_gpu_cached: bool = False,
        allocation_tag: int = 0xCAFE,
    ) -> GpuMemory:
        aligned_size = align_up(size, self.PAGE_SIZE)

        nvmap_handle = self.nvmap.create(aligned_size)
        nvmap_fd = self.nvmap.get_fd(nvmap_handle)

        NVMAP_HANDLE_UNCACHEABLE: int = 0 << 0
        NVMAP_HANDLE_WRITE_COMBINE: int = 1 << 0

        flags = NVMAP_HANDLE_UNCACHEABLE

        if is_cpu_cached:
            flags = NVMAP_HANDLE_WRITE_COMBINE

        try:
            self.nvmap.allocate(
                nvmap_handle, 1, flags | (allocation_tag << 16), self.PAGE_SIZE
            )

            gpu_flags = NVGPU_AS_MAP_BUFFER_FLAGS_DIRECT_KIND_CTRL

            if is_gpu_cached:
                gpu_flags |= NVGPU_AS_MAP_BUFFER_FLAGS_CACHEABLE

            (_, _, gpu_address) = self.address_space.map_buffer_ex(
                nvmap_fd, gpu_flags, self.PAGE_SIZE, 0, 0
            )
        except ErrnoException as e:
            self.nvmap.free(nvmap_handle)

            raise e

        mapping: mmap = mmap(nvmap_fd, aligned_size, MAP_SHARED, PROT_READ | PROT_WRITE)

        return GpuMemory(
            mapping,
            self.nvmap,
            nvmap_handle,
            size,
            self.address_space,
            gpu_address,
            aligned_size,
        )

    def submit_commands(
        self,
        command_buffers: List[CommandBuffer],
        external_wait: Optional[SubmittedCommandBuffer] = None,
    ) -> List[SubmittedCommandBuffer]:
        command_buffers_gpu_memory: List[GpuMemory] = list()

        for command_buffer in command_buffers:
            memory = self.create_gpu_memory(len(command_buffer.buffer))
            memory[0 : len(command_buffer.buffer)] = command_buffer.buffer
            command_buffers_gpu_memory.append(memory)

        user_queue: List[c_ulong] = list()

        for gpu_memory in command_buffers_gpu_memory:
            user_queue.append(
                c_ulong(gpu_memory.gpu_address | (gpu_memory.user_size // 4) << 42)
            )

        flags = (
            NVGPU_SUBMIT_GPFIFO_FLAGS_SYNC_FENCE | NVGPU_SUBMIT_GPFIFO_FLAGS_FENCE_GET
        )
        waiting_fence = None

        if external_wait is not None:
            waiting_fence = nvgpu_fence()
            waiting_fence.id = external_wait.external_wait_fd

            flags |= NVGPU_SUBMIT_GPFIFO_FLAGS_FENCE_WAIT

        result = self.channel.submit_gpfifo(user_queue, flags, waiting_fence)

        assert result is not None

        external_wait_fd = result.id
        external_wait_poll = poll()
        external_wait_poll.register(external_wait_fd, POLLOUT | POLLIN)

        submitted_command_buffers: List[SubmittedCommandBuffer] = list()

        for gpu_memory in command_buffers_gpu_memory:
            submitted_command_buffers.append(
                SubmittedCommandBuffer(gpu_memory, external_wait_fd, external_wait_poll)
            )

        return submitted_command_buffers

    def submit_command(
        self,
        command_buffer: CommandBuffer,
        external_wait: Optional[SubmittedCommandBuffer] = None,
    ) -> SubmittedCommandBuffer:
        return self.submit_commands([command_buffer], external_wait)[0]
