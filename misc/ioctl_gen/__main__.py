import clang
import clang.cindex as cl
from clang.cindex import TranslationUnit
import sys

from . import *

LIBCLANG_PATH = "/usr/lib/aarch64-linux-gnu/libclang-13.so"

# "/usr/src/linux-headers-4.9.201-tegra-ubuntu18.04_aarch64/nvgpu/include/uapi/linux/nvgpu.h"
# "/usr/src/linux-headers-4.9.201-tegra-ubuntu18.04_aarch64/nvidia/include/uapi/linux/nvmap.h"
target_header = sys.argv[1]

cl.Config.set_library_file(LIBCLANG_PATH)
idx = clang.cindex.Index.create()
tu = idx.parse(target_header, options=TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD)
parse_header(target_header, tu.cursor)
