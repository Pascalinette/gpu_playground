import re
import sys
import ctypes

from ada.hw.compute_a_qmd import *


regex = r"\.VALUE\ \=\ (?P<raw_data>.*)\n"

matches = re.finditer(regex, open(sys.argv[1], "r").read(), re.MULTILINE)

raw_bin = bytearray()

for matchNum, match in enumerate(matches, start=1):
    raw_val = match.group('raw_data')
    val = int(raw_val, 16)

    raw_bin += val.to_bytes(4, byteorder='little')

with open(sys.argv[2], "wb") as f:
    f.write(raw_bin)

qmd_struct = qmdv0300.from_buffer_copy(raw_bin)

print("QMD output")
for raw_field in qmd_struct._fields_:
    field_name = raw_field[0]
    print("    {}: {}".format(field_name, getattr(qmd_struct, field_name)))
