from argparse import *
from . import *

parser = ArgumentParser(description="Fuzz IPA/SM/SPA ISA")
parser.add_argument(
    "operation", help="The operation to perform (generate_base_definition for example)"
)
parser.add_argument(
    "input_argument",
    help="Input argument for the operation (sm_version for generate_base_definition)",
)
parser.add_argument("output_file", help="Output File")
parser.add_argument(
    "--threads_count", type=int, help="Number of threads to use to fuzz", default=1
)

args = parser.parse_args()

if args.operation == "generate_base_definition":
    generate_base_definition(
        args.output_file, args.input_argument.upper(), args.threads_count
    )
elif args.operation == "custom":
    custom(args.input_argument, args.output_file, args.threads_count)
else:
    raise Exception(args.operation)
