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


def split_instruction(instruction_raw: str) -> List[str]:
    raw_inst_components = instruction_raw.split(" ")

    # Remove usual final semicolon
    raw_inst_components[-1] = raw_inst_components[-1][0:-1]

    return raw_inst_components


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

    # Ensure all instructions entries are sorted
    for instruction in raw_definition:
        raw_definition[instruction] = sorted(
            raw_definition[instruction], key=lambda entry: entry["value"]
        )

    for instruction in raw_definition:
        result = dumb_instruction_deduplicate(
            file_name, instruction, raw_definition[instruction], sm_version
        )
        if result is not None:
            raw_definition[instruction] = result

    for instruction in raw_definition:
        result = instruction_deduplicate_arguments(raw_definition[instruction])
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
def dumb_instruction_deduplicate(
    file_name: str, instruction_name: str, data: list, sm_version: str
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
            file_name, sm_version, cleared_instruction_value
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


def diff_parsed_instruction_by_part(a: List[dict], b: List[dict], key: str) -> bool:
    if len(a) != len(b):
        return False

    for index in range(len(a)):
        if a[index][key] != b[index][key]:
            return False

    return True


def instruction_deduplicate_arguments(data: list) -> Optional[list]:
    # If there is only one variant, no deduplication needed
    if len(data) == 1:
        return None

    result = []

    already_scanned = []

    for a_instruction_entry_index in range(len(data)):
        a_instruction_entry_parsed = parse_disassembly(
            data[a_instruction_entry_index]["disassembly"]
        )
        a_mnemonic_entry = get_instruction_part_by_type(
            a_instruction_entry_parsed, "mnemonic"
        )

        if a_mnemonic_entry["value"] in already_scanned:
            continue

        temp_list = []
        is_valid = True

        for b_instruction_entry_index in range(
            a_instruction_entry_index + 1, len(data)
        ):
            b_instruction_entry_parsed = parse_disassembly(
                data[b_instruction_entry_index]["disassembly"]
            )
            b_mnemonic_entry = get_instruction_part_by_type(
                b_instruction_entry_parsed, "mnemonic"
            )

            if a_mnemonic_entry["value"] == b_mnemonic_entry["value"]:
                temp_list.append(data[b_instruction_entry_index])

                if not diff_parsed_instruction_by_part(
                    a_instruction_entry_parsed, b_instruction_entry_parsed, "type"
                ):
                    is_valid = False
                    break

        result.append(data[a_instruction_entry_index])

        if not is_valid:
            # Add back normal stuffs
            result.extend(temp_list)

        already_scanned.append(a_mnemonic_entry["value"])

    return result


def get_instruction_argument_type(raw_part: str) -> str:
    data_type = "unknown"
    start_index = 0

    if raw_part[0] == "-":
        start_index = 1

    if raw_part[start_index] == "R":
        if raw_part.find("+") != -1:
            data_type = "register_with_immediate_integer"
        else:
            data_type = "register"
    if raw_part[0] == "P":
        data_type = "predicate_register"
    elif raw_part[start_index : start_index + 2] == "0x":
        data_type = "immediate_integer"
    elif raw_part[start_index : start_index + 2] == "c[":
        data_type = "const_memory_addr"
    elif raw_part[start_index : start_index + 2] == "a[":
        data_type = "attribute"
    elif raw_part[start_index] == "[" and raw_part[-1] == "]":
        sub_part = get_instruction_argument_type(raw_part[start_index + 1 : -1])

        if sub_part != "unknown":
            data_type = "memory_address_from_" + sub_part
    elif raw_part[start_index] == "|" and raw_part[-1] == "|":
        sub_part = get_instruction_argument_type(raw_part[start_index + 1 : -1])

        if sub_part != "unknown":
            data_type = "absolute_value_from_" + sub_part
    # TODO: do something about .NEG syntax???
    else:
        # Attempt parsing as integer
        try:
            int(raw_part)

            data_type = "immediate_integer"
        except ValueError:
            # if not valid, attempt parsing as float
            try:
                float(raw_part)

                data_type = "immediate_float"
            except ValueError:
                pass

    return data_type


def parse_disassembly(disassembly: str) -> list:
    result = []

    instruction_split = split_instruction(disassembly)

    skip_index = 0

    if instruction_split[0][0] == "@":
        result.append(
            {"index": 0, "type": "cond_predicate", "value": instruction_split[0][1:]}
        )
        skip_index = 1

    instruction_split = instruction_split[skip_index:]

    result.append(
        {"index": skip_index, "type": "mnemonic", "value": instruction_split[0]}
    )

    instruction_split = instruction_split[1:]

    for raw_part_index in range(len(instruction_split)):
        raw_part = instruction_split[raw_part_index]

        if len(raw_part) == 0 or (len(raw_part) == 1 and raw_part[0] == ","):
            continue

        if raw_part[-1] == ",":
            raw_part = raw_part[0:-1]

        data_type = get_instruction_argument_type(raw_part)

        data = {
            "index": skip_index + 1 + raw_part_index,
            "type": data_type,
            "value": raw_part,
        }

        if data["type"] == "unknown":
            pass
            # print(disassembly)
            # print(data)

            # raise Exception((data, disassembly))

        result.append(data)

    return result


def get_instruction_part_by_type(
    parsed_instruction: list, part_type: str
) -> Optional[dict]:
    for part in parsed_instruction:
        if part["type"] == part_type:
            return part

    return None


def get_instruction_part_by_index(
    parsed_instruction: list, part_index: int
) -> Optional[dict]:
    for part in parsed_instruction:
        if part["index"] == part_index:
            return part

    return None


def parse_instruction_name(instruction_raw: str) -> str:
    parsed_instruction = parse_disassembly(instruction_raw)

    instruction_with_modifiers_def = parsed_instruction[0]

    if instruction_with_modifiers_def["type"] == "cond_predicate":
        instruction_with_modifiers_def = parsed_instruction[1]

    # Finally only grab the base name
    instruction = instruction_with_modifiers_def["value"].split(".")[0]

    return instruction


def search_register_bits(
    file_name: str, instruction_value: int, register_part: dict, sm_version: str
) -> Optional[dict]:
    sm_description = get_sm_desc(sm_version)
    register_count = sm_description["register_count"]
    opcode_start = sm_description["opcode_start"]

    # FIXME: Terrible I know
    register_bits_count = len(bin(register_count)) - 2

    max_bits = opcode_start - register_bits_count
    target_index = register_part["index"]

    # We do not want to hit RZ
    register_value = register_count - 1
    expected_string_name = f"R{register_value}"

    for bit_index in range(max_bits):
        instruction = instruction_value | register_value << bit_index

        (success, result) = test_instruction(file_name, sm_version, instruction)

        if success:
            (_, raw_instruction, _) = result[1]

            parsed_instruction = parse_disassembly(raw_instruction)

            parsed_register = get_instruction_part_by_index(
                parsed_instruction, target_index
            )

            if (
                parsed_register is not None
                and parsed_register["type"] == register_part["type"]
                and parsed_register["value"] == expected_string_name
            ):
                return {
                    "index": parsed_register["index"],
                    "type": parsed_register["type"],
                    "bit_mask": register_count << bit_index,
                }

    return None


def custom(input_argument: str, output_file: str, threads_count: int):
    sm_version = "SM53"
    sm_desc = get_sm_desc(sm_version)

    assert sm_desc is not None

    (fd, file_name) = tempfile.mkstemp()
    os.close(fd)

    with open(input_argument, "r") as f:
        raw_definition = json.loads(f.read())

    new_definition = {}

    for instruction in raw_definition:
        result = []

        for instruction_definition in raw_definition[instruction]:
            parsed_instruction = parse_disassembly(
                instruction_definition["disassembly"]
            )

            mnemonic = get_instruction_part_by_type(parsed_instruction, "mnemonic")
            print(mnemonic)

            fields = []

            for parsed_part in parsed_instruction:
                if parsed_part["type"] == "register":
                    output = search_register_bits(
                        file_name,
                        instruction_definition["value"],
                        parsed_part,
                        sm_version,
                    )

                    if output is not None:
                        fields.append(output)

            result.append(
                {
                    "opcode": instruction_definition["value"],
                    "mnemonic": mnemonic["value"],
                    "disassembly": instruction_definition["disassembly"],
                    "fields": fields,
                }
            )

        new_definition[instruction] = result

    with open(output_file, "w") as f:
        f.write(json.dumps(new_definition, indent=4))

    os.unlink(file_name)
    sys.exit(0)
