import clang
import clang.cindex as cl
from clang.cindex import TranslationUnit
import sys

from argparse import *

from . import *

c = "/usr/lib/aarch64-linux-gnu/libclang-13.so"

parser = ArgumentParser(description="Process a C header to Python.")
parser.add_argument("target_header", help="The C header")
parser.add_argument("output_file", help="The python generated file")
parser.add_argument(
    "--libclang_path", default=None, help="The search path for libclang.so"
)

args = parser.parse_args()


# "/usr/src/linux-headers-4.9.201-tegra-ubuntu18.04_aarch64/nvgpu/include/uapi/linux/nvgpu.h"
# "/usr/src/linux-headers-4.9.201-tegra-ubuntu18.04_aarch64/nvidia/include/uapi/linux/nvmap.h"
target_header = args

if args.libclang_path is not None:
    cl.Config.set_library_file(args.libclang_path)

idx = clang.cindex.Index.create()
tu = idx.parse(
    args.target_header, options=TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
)
parse_header(args.target_header, args.output_file, tu.cursor)
