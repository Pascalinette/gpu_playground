from typing import Dict, Optional
from drm_header import *
from utils import BlockDevice


class DrmCard(BlockDevice):
    def __init__(self, path: Optional[str] = "/dev/dri/card0", fd: int = -1) -> None:
        super().__init__(path, fd)

    def get_version(self) -> Dict[str, Any]:
        # First we grab the size of the strings
        request = drm_version()
        self.check_result(drm_get_version(self.fd, request))

        # Now we perform the right request
        request.name = create_string_buffer(request.name_len)
        request.date = create_string_buffer(request.date_len)
        request.desc = create_string_buffer(request.desc_len)
        self.check_result(drm_get_version(self.fd, request))

        version_major = int(request.version_major)
        version_minor = int(request.version_minor)
        version_patchlevel = int(request.version_patchlevel)
        version = version_major << 24 | version_minor << 8 | version_patchlevel
        name = request.name[: request.name_len].decode("utf-8")
        date = request.date[: request.date_len].decode("utf-8")
        desc = request.desc[: request.desc_len].decode("utf-8")

        return {
            "version": version,
            "version_major": version_major,
            "version_minor": version_minor,
            "version_patchlevel": version_patchlevel,
            "name": name,
            "date": date,
            "desc": desc,
        }
