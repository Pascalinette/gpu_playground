from difflib import diff_bytes
import json
import os
from subprocess import STDOUT
import struct
import sys
import subprocess
import tempfile
import threading
from typing import Dict, List, Optional, Tuple, Union

SM_DESCRIPTION = {
    "SM35": {
        "arch": "kepler",
        "register_count": 255,
        "predicate_count": 7,
        "instruction_bits": 64,
        "opcode_start": 54,
        "opcode_end": 64,
        # control / immediate / standard
        "opcode_extra_base": [0x0, 0x1, 0x2],
        "control_bits": 64,
        "control_instruction_count": 7,
        "stub_scheduler_ctl": 0x081F8000FC0007E0,
        "stub_instruction": 0x18000000001C003C,
    },
    "SM5x": {
        "arch": "maxwell",
        "register_count": 255,
        "predicate_count": 7,
        "instruction_bits": 64,
        "opcode_start": 50,
        "opcode_end": 64,
        "opcode_extra_base": [0x0],
        "control_bits": 64,
        "control_instruction_count": 3,
        "stub_scheduler_ctl": 0x001F8000FC0007E0,
        "stub_instruction": 0x50B0000000070F00,
    },
    # NOTE: Pascal is just Maxwell gen 3
    "SM6x": {
        "arch": "pascal",
        "register_count": 255,
        "predicate_count": 7,
        "instruction_bits": 64,
        "opcode_start": 50,
        "opcode_end": 64,
        "opcode_extra_base": [0x0],
        "control_bits": 64,
        "control_instruction_count": 3,
        "stub_scheduler_ctl": 0x001F8000FC0007E0,
        "stub_instruction": 0x50B0000000070F00,
    },
    # TODO: Volta, Turing, Ampere
}


SM_DESCRIPTION["SM50"] = SM_DESCRIPTION["SM5x"]
SM_DESCRIPTION["SM51"] = SM_DESCRIPTION["SM5x"]
SM_DESCRIPTION["SM52"] = SM_DESCRIPTION["SM5x"]
SM_DESCRIPTION["SM53"] = SM_DESCRIPTION["SM5x"]

SM_DESCRIPTION["SM60"] = SM_DESCRIPTION["SM6x"]
SM_DESCRIPTION["SM61"] = SM_DESCRIPTION["SM6x"]
SM_DESCRIPTION["SM62"] = SM_DESCRIPTION["SM6x"]


# TODO: Handle volta+ changes somehow
def generate_test_binary(sm_version: str, instruction: int) -> bytearray:
    sm_desc = get_sm_desc(sm_version)

    assert sm_desc is not None

    scheduler_ctrl_encoding_type = None
    instruction_encoding_type = None

    stub_scheduler_ctrl = sm_desc["stub_scheduler_ctl"]
    stub_instruction = sm_desc["stub_instruction"]

    if sm_desc["instruction_bits"] == 64:
        instruction_encoding_type = "Q"
    elif sm_desc["instruction_bits"] == 32:
        instruction_encoding_type = "I"
    else:
        raise Exception(sm_desc["instruction_bits"])

    if sm_desc["control_bits"] == 64:
        scheduler_ctrl_encoding_type = "Q"
    elif sm_desc["control_bits"] == 32:
        scheduler_ctrl_encoding_type = "I"
    else:
        raise Exception(sm_desc["control_bits"])

    result = bytearray()
    result += bytearray(struct.pack(scheduler_ctrl_encoding_type, stub_scheduler_ctrl))
    result += bytearray(struct.pack(instruction_encoding_type, instruction))

    for _ in range(sm_desc["control_instruction_count"] - 1):
        result += bytearray(struct.pack(instruction_encoding_type, stub_instruction))

    return result


def parse_nvdis_output(raw_output: bytes) -> List[Tuple[int, str, int]]:
    result: List[Tuple[int, str, int]] = list()

    raw_output_str = raw_output.decode("utf-8")

    lines = list(filter(None, raw_output_str.split("\n")))[1:]

    for line in lines:
        line = line.strip()

        splitted_line = list(filter(None, line.split(" ")))

        # Parse the special case of the scheduler flag
        if (
            len(splitted_line) == 3
            and splitted_line[0] == "/*"
            and splitted_line[2] == "*/"
        ):
            raw_value = int(splitted_line[1], 16)

            result.append((-1, "sched", raw_value))
        else:
            binary_offset = int("0x" + splitted_line[0][2:-2], 16)
            raw_value = int(splitted_line[-2], 16)

            instruction_raw = " ".join(splitted_line[1:-3])

            result.append((binary_offset, instruction_raw, raw_value))

    return result


def parse_instruction_name(instruction_raw: str) -> str:
    raw_inst_components = instruction_raw.split(" ")

    instruction_with_modifiers = raw_inst_components[0]

    # Exclude predicate modifier that could be before the instruction name
    if instruction_with_modifiers[0] == "@":
        instruction_with_modifiers = raw_inst_components[1]

    # Finally only grab the base name
    instruction = instruction_with_modifiers.split(".")[0]

    # Remove usual final semicolon
    if instruction[-1] == ";":
        instruction = instruction[0:-1]

    return instruction


def execute_disassembler(
    sm_version: str, path: str
) -> Tuple[bool, Union[str, List[Tuple[int, str, int]]]]:
    try:
        result = subprocess.check_output(
            args=["nvdisasm", "-ndf", "-hex", "--binary", sm_version, path],
            stderr=STDOUT,
        )

        result = parse_nvdis_output(result)

        return (True, result)
    except subprocess.CalledProcessError as e:
        return (False, e.output.decode("utf-8"))


def test_instruction(
    file_name: str, sm_version: str, instruction: int
) -> Tuple[bool, Union[str, List[Tuple[int, str, int]]]]:
    with open(file_name, "wb") as f:
        f.write(generate_test_binary(sm_version, instruction))

    result = execute_disassembler(sm_version, file_name)

    return result


def create_instruction_entry(disassembly: str, value: int) -> dict:
    return {
        "disassembly": disassembly,
        "value": value,
    }


def get_sm_desc(sm_version: str) -> Optional[dict]:
    if sm_version in SM_DESCRIPTION:
        return SM_DESCRIPTION[sm_version]

    return None


def fuzz_instruction_range(
    thread_id: int,
    result_dict: dict,
    sm_version: str,
    opcode_extra_base: List[int],
    range_start: int,
    range_end: int,
    bit_shift: int,
):
    (fd, file_name) = tempfile.mkstemp()
    os.close(fd)

    pass_count = range_end - range_start

    print((thread_id, range_start, range_end))

    for extra_base_index in range(len(opcode_extra_base)):
        extra_base = opcode_extra_base[extra_base_index]

        for possible_op_code in range(range_start, range_end):
            instruction = possible_op_code << bit_shift | extra_base

            (success, result) = test_instruction(file_name, sm_version, instruction)

            if success:
                (_, raw_instruction, _) = result[1]

                instruction_name = parse_instruction_name(raw_instruction)

                if not instruction_name in result_dict:
                    result_dict[instruction_name] = list()

                result_dict[instruction_name].append(
                    create_instruction_entry(raw_instruction, instruction)
                )
                # print(f"{thread_id}: {instruction:064b}: SUCCESS ({instruction_name})")

            else:
                pass
                # result = result.strip()

                # print(f"{instruction:064b}: FAIL ({result[0:-1]})")

            if (possible_op_code % 100) == 0:
                percent = (
                    ((possible_op_code - range_start) + extra_base_index * pass_count)
                    / (pass_count * len(opcode_extra_base))
                ) * 100
                print(f"{thread_id}: {percent} %")

    os.unlink(file_name)


def generate_base_definition(output_file: str, sm_version: str, threads_count: int):
    sm_desc = get_sm_desc(sm_version)

    assert sm_desc is not None

    (fd, file_name) = tempfile.mkstemp()
    os.close(fd)

    bit_start = sm_desc["opcode_start"]
    bit_end = sm_desc["opcode_end"]
    bit_count = bit_end - bit_start

    opcode_extra_base = sm_desc["opcode_extra_base"]

    print(f"Possible opcode range: {bit_start}:{bit_end}")

    max_value = (1 << bit_count) - 1

    print(f"Need to test {(len(opcode_extra_base) * max_value) + 1} possibilities")
    print(f"{threads_count}")

    amount_to_process_per_thread = max_value // threads_count

    workers: List[threading.Thread] = list()
    workers_result = list()

    for _ in range(threads_count):
        workers_result.append(dict())

    for thread_id in range(threads_count):
        thread = threading.Thread(
            target=fuzz_instruction_range,
            args=(
                thread_id,
                workers_result[thread_id],
                sm_version,
                opcode_extra_base,
                thread_id * amount_to_process_per_thread,
                (thread_id + 1) * amount_to_process_per_thread,
                bit_start,
            ),
        )

        workers.append(thread)

        thread.start()

    for worker in workers:
        worker.join()

    raw_definition = dict()
    for entry in workers_result:
        for key in entry:
            if not key in raw_definition:
                raw_definition[key] = list()

            raw_definition[key].extend(entry[key])

    for instruction in raw_definition:
        result = dumb__instruction_deduplicate(
            file_name, instruction, raw_definition[instruction]
        )
        if result is not None:
            raw_definition[instruction] = result

    with open(output_file, "w") as f:
        f.write(json.dumps(raw_definition, indent=4))

    sys.exit(0)


def bit_diff(a: int, b: int, bits_len: int = 64) -> List[Tuple[int, int, int]]:
    result: List[Tuple[int, int, int]] = list()

    # If equal well there is no difference.
    if a == b:
        return result

    for bit_index in range(bits_len):
        a_value = (a >> bit_index) & 1
        b_value = (b >> bit_index) & 1

        if a_value != b_value:
            result.append((bit_index, a_value, b_value))

    return result


# TODO: group by disassembly result and do bit flipping according to that (to avoid possibly loosing mirrors)
def dumb__instruction_deduplicate(
    file_name: str, instruction_name: str, data: list
) -> Optional[list]:
    # If there is only one variant, no deduplication needed
    if len(data) == 1:
        return None

    # Let's try simple dumb deduplication
    # If we get the same instruction while clearning all different bits, this is probably our target.
    first_instruction_entry = data[0]
    first_instruction_entry_value = first_instruction_entry["value"]

    bit_differences = dict()

    for instruction_entry in data[1:]:
        instruction_value = instruction_entry["value"]

        bit_differences[instruction_value] = bit_diff(
            first_instruction_entry_value, instruction_value, 64
        )

    cleared_instruction_value = first_instruction_entry_value

    for index in bit_differences:
        for (bit_index, _, _) in bit_differences[index]:
            cleared_instruction_value = cleared_instruction_value & ~(1 << bit_index)

    if cleared_instruction_value != first_instruction_entry_value:
        (instruction_valid, raw_output) = test_instruction(
            file_name, "SM53", cleared_instruction_value
        )

        if not instruction_valid:
            return None

        (_, raw_instruction, _) = raw_output[1]

        generated_instruction_name = parse_instruction_name(raw_instruction)

        if generated_instruction_name != instruction_name:
            return None

        return [create_instruction_entry(raw_instruction, cleared_instruction_value)]
    else:
        return [first_instruction_entry]


def custom(input_argument: str, output_file: str, threads_count: int):
    (fd, file_name) = tempfile.mkstemp()
    os.close(fd)

    print(test_instruction(file_name, "SM53", int(input_argument[2:], 16)))

    os.unlink(file_name)
    sys.exit(0)
