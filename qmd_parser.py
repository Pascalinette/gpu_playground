import re
import sys
import ctypes

from ada.hw.compute_a_qmd import *

# For raw QMD load
# regex = r"\.V\ \=\ \((?P<raw_data>.*)\)\n"
regex = r"\.VALUE\ \=\ (?P<raw_data>.*)\n"

matches = re.finditer(regex, open(sys.argv[1], "r").read(), re.MULTILINE)

raw_bin = bytearray()

for matchNum, match in enumerate(matches):\
    # Skip the first 3 dwords (args of the MME macro)
    if matchNum < 3:
        continue
    raw_val = match.group('raw_data')
    val = int(raw_val, 16)
    raw_bin += val.to_bytes(4, byteorder='little')

assert len(raw_bin) == 256

with open(sys.argv[2], "wb") as f:
    f.write(raw_bin)

qmd_struct = qmdv0300.from_buffer_copy(raw_bin)

print("QMD output")
for raw_field in qmd_struct._fields_:
    field_name = raw_field[0]
    if field_name.startswith("pad_"):
        continue

    print("    {}: {}".format(field_name, getattr(qmd_struct, field_name)))
