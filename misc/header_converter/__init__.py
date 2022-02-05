from datetime import datetime
import os
import re
from typing import Iterable, List, Optional, Tuple, Union
from clang.cindex import Cursor, CursorKind, TokenKind


class CodeStream(object):
    def __init__(self):
        self.code: str = ""
        self.spaces: int = 0

    def get(self) -> str:
        return self.code

    def indent(self) -> None:
        self.spaces += 4

    def unindent(self) -> None:
        self.spaces -= 4

    def write(self, text: str) -> None:
        self.code += text

    def begin_line(self) -> None:
        self.write(" " * self.spaces)

    def write_line(self, line: str = "") -> None:
        self.begin_line()
        self.write(line + "\n")


def create_codestream(start_time: datetime):
    stream = CodeStream()

    stream.write_line("# AUTOGENERATED: DO NOT EDIT")
    stream.write_line("# Last update date: {0}".format(start_time))
    stream.write_line()

    return stream


def print_node(node):
    try:
        print((node.kind, node.spelling, node.extent, node.result_type.spelling))
    except ValueError:
        # some kinds aren't defined
        print((node._kind_id, node.spelling, node.extent))


def print_recursive_childs(node):
    print("Childs:")
    for node in node.get_children():
        print_node(node)
        print_recursive_childs(node)


def print_arguments(node):
    print("Arguments:")
    for node in node.get_arguments():
        print_node(node)


class ParsedStructField(object):
    name: str
    field_type: Union[str, "ParsedStruct", "ParsedUnion"]
    element_count: Optional[int]

    def __init__(
        self,
        name: str,
        field_type: Union[str, "ParsedStruct", "ParsedUnion"],
        element_count: Optional[int] = None,
    ) -> None:
        self.name = name
        self.field_type = field_type
        self.element_count = element_count

    def __repr__(self) -> str:
        return f'ParsedStructField(name="{self.name}", field_type="{self.field_type}")'


type_mapping = {
    "__s8": "c_byte",
    "__u8": "c_ubyte",
    "__s16": "c_short",
    "__u16": "c_ushort",
    "__s32": "c_int",
    "__u32": "c_uint",
    "__s64": "c_long",
    "__u64": "c_ulong",
    "short": "c_short",
    "unsigned short": "c_ushort",
    "int": "c_int",
    "unsigned int": "c_uint",
    "long": "c_long",
    "unsigned long": "c_ulong",
}


def get_python_ctype_type(field: ParsedStructField) -> Optional[str]:
    if type(field.field_type) is str:
        field_type_name = field.field_type

        if field_type_name.startswith("struct"):
            return field_type_name[len("struct") + 1 :]

        # Remove volatile because we cannot represent that.
        if field_type_name.startswith("volatile"):
            field_type_name = field_type_name[len("volatile") + 1 :]

        # TODO pointer of pointers
        if field_type_name[-1] == "*":
            adjusted_field_type_name = field_type_name[:-1].strip()

            assert adjusted_field_type_name.find("*") == -1

            if adjusted_field_type_name in type_mapping:
                return f"POINTER({type_mapping[adjusted_field_type_name]})"

        if field_type_name in type_mapping:
            return type_mapping[field_type_name]

        if field_type_name in type_mapping:
            return type_mapping[field_type_name]
    else:
        raise Exception(f"TODO {type(field.field_type)}")

    return None


class ParsedStruct(object):
    name: Optional[str]
    fields: List[ParsedStructField]
    is_packed: bool

    def __init__(
        self, name: Optional[str], fields: List[ParsedStructField], is_packed: bool
    ) -> None:
        self.name = name
        self.fields = fields
        self.is_packed = is_packed

    def __repr__(self) -> str:
        return f'ParsedStruct(name="{self.name}", fields="{self.fields}", is_packed={self.is_packed})'


class ParsedUnion(object):
    name: Optional[str]
    fields: List[ParsedStructField]

    def __init__(self, name: Optional[str], fields: List[ParsedStructField]) -> None:
        self.name = name
        self.fields = fields


def create_union(node: Cursor, is_anonymous: bool = False) -> Optional[ParsedUnion]:
    fields: List[ParsedStructField] = []
    name: Optional[str] = None
    name_is_in_next_child: bool = False

    if not is_anonymous:
        name = node.spelling

    for child in node.get_children():
        if child.kind == CursorKind.FIELD_DECL:
            result_field: ParsedStructField
            node_type = child.type.spelling
            if "[" in node_type:
                array_type = node_type[: node_type.index(" [")]
                element_count = int(
                    node_type[node_type.index(" [") + 2 : node_type.index("]")]
                )

                result_field = ParsedStructField(
                    child.spelling, array_type, element_count
                )
            else:
                result_field = ParsedStructField(child.spelling, node_type)

            if name_is_in_next_child and "unnamed" in node_type:
                result_field.field_type = fields[-1].field_type
                fields[-1] = result_field
            else:
                fields.append(result_field)

        elif child.kind == CursorKind.UNION_DECL:
            anonymous_obj = create_union(child, True)

            field_name = child.spelling
            if field_name == "":
                name_is_in_next_child = True

            assert anonymous_obj is not None

            fields.append(ParsedStructField(child.spelling, anonymous_obj))
            continue
        elif child.kind == CursorKind.STRUCT_DECL:
            anonymous_obj = create_struct(child, True)

            field_name = child.spelling
            if field_name == "":
                name_is_in_next_child = True

            assert anonymous_obj is not None

            fields.append(ParsedStructField(child.spelling, anonymous_obj))
            continue
        else:
            raise Exception("TODO " + str(child.kind))

        name_is_in_next_child = False

    return ParsedUnion(name, fields)


def create_struct(node: Cursor, is_anonymous: bool = False) -> Optional[ParsedStruct]:
    fields: List[ParsedStructField] = []
    is_packed: bool = False
    name: Optional[str] = None
    name_is_in_next_child: bool = False

    if not is_anonymous:
        name = node.spelling

    for child in node.get_children():
        if child.kind == CursorKind.FIELD_DECL:
            result_field: ParsedStructField
            node_type = child.type.spelling
            if "[" in node_type:
                array_type = node_type[: node_type.index(" [")]
                element_count = int(
                    node_type[node_type.index(" [") + 2 : node_type.index("]")]
                )

                result_field = ParsedStructField(
                    child.spelling, array_type, element_count
                )
            else:
                result_field = ParsedStructField(child.spelling, node_type)

            if name_is_in_next_child and "unnamed" in node_type:
                result_field.field_type = fields[-1].field_type
                fields[-1] = result_field
            else:
                fields.append(result_field)

        elif child.kind == CursorKind.PACKED_ATTR:
            is_packed = True
        elif child.kind == CursorKind.UNION_DECL:
            anonymous_obj = create_union(child, True)

            field_name = child.spelling
            if field_name == "":
                name_is_in_next_child = True

            assert anonymous_obj is not None

            fields.append(ParsedStructField(child.spelling, anonymous_obj))
            continue
        elif child.kind == CursorKind.STRUCT_DECL:
            anonymous_obj = create_struct(child, True)

            field_name = child.spelling
            if field_name == "":
                name_is_in_next_child = True

            assert anonymous_obj is not None

            fields.append(ParsedStructField(child.spelling, anonymous_obj))
            continue
        else:
            raise Exception("TODO " + str(child.kind))

        name_is_in_next_child = False

    return ParsedStruct(name, fields, is_packed)


def print_union(stream: CodeStream, struct: ParsedUnion) -> None:
    assert struct.name is not None

    for index in range(len(struct.fields)):
        field = struct.fields[index]
        if type(field.field_type) is ParsedStruct:
            assert field.field_type.name is None

            field.field_type.name = struct.name + "_unamed_struct_" + str(index)

            print_struct(stream, field.field_type)
        elif type(field.field_type) is ParsedUnion:
            assert field.field_type.name is None

            field.field_type.name = struct.name + "_unamed_union_" + str(index)
            print_union(stream, field.field_type)

    ctypes_fields: List[Tuple[str, str]] = list()

    for field in struct.fields:
        if type(field.field_type) is str:
            ctype_field_type = get_python_ctype_type(field)

            if ctype_field_type is None:
                print(
                    f"Warning {field.name} ({field.field_type}) type not found, assuming 1:1 mapping..."
                )
                ctype_field_type = field.field_type

            if field.element_count is not None:
                ctype_field_type = f"{ctype_field_type} * {field.element_count}"

            ctypes_fields.append((field.name, ctype_field_type))
        elif type(field.field_type) in [ParsedStruct, ParsedUnion]:
            ctypes_fields.append((field.name, field.field_type.name))

    stream.write_line(f"class {struct.name}(Union):")
    stream.indent()
    stream.write_line("_fields_ = [")
    stream.indent()

    unamed_index = 0

    for (field_name, field_type) in ctypes_fields:
        if field_name == "":
            field_name = f"unamed_field{unamed_index}"
            unamed_index += 1

        stream.write_line(f'("{field_name}", {field_type}),')

    stream.unindent()
    stream.write_line("]")
    stream.unindent()
    stream.write_line()
    stream.write_line()


def print_struct(stream: CodeStream, struct: ParsedStruct) -> None:
    assert struct.name is not None

    for index in range(len(struct.fields)):
        field = struct.fields[index]
        if type(field.field_type) is ParsedStruct:
            assert field.field_type.name is None

            field.field_type.name = struct.name + "_unamed_struct_" + str(index)

            print_struct(stream, field.field_type)
        elif type(field.field_type) is ParsedUnion:
            assert field.field_type.name is None

            field.field_type.name = struct.name + "_unamed_union_" + str(index)
            print_union(stream, field.field_type)

    ctypes_fields: List[Tuple[str, str]] = list()

    for field in struct.fields:
        if type(field.field_type) is str:
            ctype_field_type = get_python_ctype_type(field)

            if ctype_field_type is None:
                print(
                    f"Warning {field.name} ({field.field_type}) type not found, assuming 1:1 mapping..."
                )
                ctype_field_type = field.field_type

            if field.element_count is not None:
                ctype_field_type = f"{ctype_field_type} * {field.element_count}"

            ctypes_fields.append((field.name, ctype_field_type))
        elif type(field.field_type) in [ParsedStruct, ParsedUnion]:
            ctypes_fields.append((field.name, field.field_type.name))

    stream.write_line(f"class {struct.name}(Structure):")
    stream.indent()

    if struct.is_packed:
        stream.write_line("_pack_ = 1")

    stream.write_line("_fields_ = [")
    stream.indent()

    unamed_index = 0

    for (field_name, field_type) in ctypes_fields:
        if field_name == "":
            field_name = f"unamed_field{unamed_index}"
            unamed_index += 1

        stream.write_line(f'("{field_name}", {field_type}),')

    stream.unindent()
    stream.write_line("]")
    stream.unindent()
    stream.write_line()
    stream.write_line()


def filter_uneeded_stuffs(target_header: str, node: Cursor) -> Iterable[Cursor]:
    result: Iterable[Cursor] = list()

    for child in node.get_children():
        child_node: Cursor = child

        if (
            child_node.kind
            in (
                CursorKind.MACRO_INSTANTIATION,
                CursorKind.MACRO_DEFINITION,
                CursorKind.STRUCT_DECL,
                CursorKind.UNION_DECL,
            )
            and child_node.extent.start.file is not None
            and child_node.extent.start.file.name == target_header
            and child_node.spelling not in ["__packed", "__user"]
        ):
            result.append(child_node)

    return result


class SimpleMacroDefinition(object):
    name: str
    value: Union[str, int]

    def __init__(self, name: str, value: Union[str, int]) -> None:
        self.name = name
        self.value = value

    def __repr__(self) -> str:
        return f'SimpleMacroDefinition(name="{self.name}", value="{self.value}")'


class NvBitfieldMacroDefinition(object):
    name: str
    offset_end: int
    offset_start: int

    def __init__(self, name: str, offset_end: int, offset_start: int) -> None:
        self.name = name
        self.offset_end = offset_end
        self.offset_start = offset_start

    def __repr__(self) -> str:
        return f'NvBitfieldMacroDefinition(name="{self.name}", offset_end="{self.offset_end}", offset_start="{self.offset_start}")'


class NvBitfieldQmdMacroDefinition(NvBitfieldMacroDefinition):
    offset_next_element: int
    element_count: int

    def __init__(
        self,
        name: str,
        offset_end: int,
        offset_start: int,
        offset_next_element: int = 0,
        element_count: int = 1,
    ) -> None:
        self.offset_next_element = offset_next_element
        self.element_count = element_count
        super().__init__(name, offset_end, offset_start)

    def __repr__(self) -> str:
        return f'NvBitfieldQmdMacroDefinition(name="{self.name}", offset_end="{self.offset_end}", offset_start="{self.offset_start}, offset_next_element="{self.offset_next_element}", element_count="{self.element_count}")'


class QmdStruct(object):
    name: str
    fields: List[Tuple[str, int, int]]

    def __init__(self, name: str) -> None:
        self.name = name
        self.fields = list()


def print_node_tokens(node: Cursor) -> None:
    tokens = list(node.get_tokens())

    tmp_str = "Token: {"

    for token in tokens:
        tmp_str += f' {token.kind}: "{token.spelling}",'

    tmp_str += " }"

    tmp_str += f" (len: {len(tokens)})"

    print(tmp_str)


def try_parse_c_integer(raw_value: str) -> Optional[int]:
    base = 10
    filter_regex = "[^0-9]"

    if raw_value.startswith("0x"):
        base = 16
        raw_value = raw_value[2:]
        filter_regex = "[^0-9a-fA-F]"

    raw_value = re.sub(filter_regex, "", raw_value)

    try:
        return int(raw_value, base)
    except ValueError:
        return None


def try_parse_c_char(raw_value: str) -> Optional[int]:
    if len(raw_value) == 3 and raw_value[0] == "'" and raw_value[2] == "'":
        return ord(raw_value[1])

    return None


def try_compute_arithmetic_operation(
    operand0: int, operation: str, operand1: int
) -> Optional[int]:
    if operand0 is None or operand1 is None:
        return None

    if operation == "+":
        return operand0 + operand1
    elif operation == "-":
        return operand0 - operand1
    elif operation == "*":
        return operand0 * operand1
    elif operation == "/":
        return operand0 // operand1
    elif operation == "%":
        return operand0 % operand1
    elif operation == "<<":
        return operand0 << operand1
    elif operation == ">>":
        return operand0 >> operand1

    return None


def try_get_constant_by_name(
    constants: List[SimpleMacroDefinition], name: str
) -> Optional[SimpleMacroDefinition]:
    for constant in constants:
        if constant.name == name:
            return constant

    return None


def try_get_type_by_name(structs: List[ParsedStruct], name: str) -> Optional[str]:
    for struct in structs:
        if struct.name == name:
            return struct.name

    return get_python_ctype_type(ParsedStructField("unknown", name))


def create_simple_macro_from_str(
    constants: List[SimpleMacroDefinition], key: str, value_node: Cursor
) -> SimpleMacroDefinition:
    value: str

    if value_node.kind == TokenKind.IDENTIFIER:
        constant = try_get_constant_by_name(constants, value_node.spelling)

        if constant is None:
            return None

        value = str(constant.value)
    else:
        value = value_node.spelling

    # Attempt to parse as an integer now
    final_value = try_parse_c_integer(value)

    if final_value is None:
        # Attempt to convert a char to int in that case
        final_value = try_parse_c_char(value)

    if final_value is None:
        final_value = value

    return SimpleMacroDefinition(key, final_value)


def try_evaluate_simple_macro(
    constants: List[SimpleMacroDefinition], node: Cursor
) -> Optional[SimpleMacroDefinition]:
    tokens = list(node.get_tokens())

    if len(tokens) < 2:
        return None

    # Remove extraneous final parentesis to simplify logics later
    if (
        tokens[1].kind == TokenKind.PUNCTUATION
        and tokens[1].spelling == "("
        and tokens[-1].kind == TokenKind.PUNCTUATION
        and tokens[-1].spelling == ")"
    ):
        tokens.pop(1)
        tokens.pop()

    macro_identifier = tokens[0]
    value0 = tokens[1]

    # If the macro is just in the "#define KEY VALUE" form create right away
    if len(tokens) == 2 and value0.kind in [TokenKind.LITERAL, TokenKind.IDENTIFIER]:
        return create_simple_macro_from_str(
            constants, macro_identifier.spelling, value0
        )

    # Try to compute macro in the form of "#define KEY OP0 << OP1"
    if len(tokens) == 4:
        operand0 = try_parse_c_integer(tokens[1].spelling)
        operation = tokens[2].spelling
        operand1 = try_parse_c_integer(tokens[3].spelling)

        res = try_compute_arithmetic_operation(operand0, operation, operand1)

        if res is not None:
            return SimpleMacroDefinition(macro_identifier.spelling, res)

    return None


class IoctlMacroDefinition(object):
    name: str
    type: int
    nr: int
    is_read: bool
    is_write: bool
    size: Optional[str]
    is_struct: bool

    def __init__(
        self,
        name: str,
        type: int,
        nr: int,
        is_read: bool,
        is_write: bool,
        size: Optional[str],
        is_struct: bool,
    ) -> None:
        self.name = name
        self.type = type
        self.nr = nr
        self.is_read = is_read
        self.is_write = is_write
        self.size = size
        self.is_struct = is_struct

    def __repr__(self) -> str:
        return f'IoctlMacroDefinition(name="{self.name}", type={self.type}, nr={self.nr}, is_read={self.is_read}, is_write={self.is_write}, size="{self.size}", is_struct="{self.is_struct}")'


def try_evaluate_ioctl_macro(
    structs: List[ParsedStruct], constants: List[SimpleMacroDefinition], node: Cursor
) -> Optional[SimpleMacroDefinition]:
    tokens = list(node.get_tokens())

    if len(tokens) < 2:
        return None

    macro_identifier = tokens[0]
    ioctl_keyword = tokens[1]

    is_read: bool = False
    is_write: bool = False
    is_struct: bool = False
    type_value: Optional[int] = None
    nr_value: Optional[int] = None
    size: Optional[str] = None

    if (
        ioctl_keyword.kind != TokenKind.IDENTIFIER
        or not ioctl_keyword.spelling.startswith("_IO")
    ):
        return None

    # Then try to remove possible "struct" keyword at position 8 if the len(tokens) == 10
    if len(tokens) == 10 and tokens[7].spelling == "struct":
        tokens.pop(7)

    if len(tokens) < 6:
        return None

    type_constant_value = try_get_constant_by_name(constants, tokens[3].spelling)
    nr_constant_value = try_get_constant_by_name(constants, tokens[5].spelling)

    if type_constant_value is not None:
        assert type(type_constant_value.value) is int

        type_value = type_constant_value.value

    if nr_constant_value is not None:
        assert type(nr_constant_value.value) is int

        nr_value = nr_constant_value.value

    if type_value is None:
        type_value = try_parse_c_integer(tokens[3].spelling)

    if nr_value is None:
        nr_value = try_parse_c_integer(tokens[5].spelling)

    if type_value is None or nr_value is None:
        return None

    if len(tokens) > 8:
        if ioctl_keyword.spelling == "_IOR":
            is_read = True
        elif ioctl_keyword.spelling == "_IOW":
            is_write = True
        elif ioctl_keyword.spelling == "_IOWR":
            is_read = True
            is_write = True

        if tokens[7].kind == TokenKind.IDENTIFIER:
            size = try_get_type_by_name(structs, tokens[7].spelling)
            is_struct = True
        elif tokens[7].kind == TokenKind.KEYWORD:
            # In case of primitive we need to concatenate tokens until the punctuation
            temp_primitive_tokens_list: List[str] = list()

            for temp_token in tokens[7:]:
                if temp_token.kind == TokenKind.PUNCTUATION:
                    break
                temp_primitive_tokens_list.append(temp_token.spelling)

            primitive_type = " ".join(temp_primitive_tokens_list)
            size = try_get_type_by_name(structs, primitive_type)

        if size is None:
            return None

    return IoctlMacroDefinition(
        macro_identifier.spelling,
        type_value,
        nr_value,
        is_read,
        is_write,
        size,
        is_struct,
    )


def try_evaluate_nv_bitfield_macro(node: Cursor) -> Optional[NvBitfieldMacroDefinition]:
    assert node.kind == CursorKind.MACRO_DEFINITION

    tokens = list(node.get_tokens())

    # Ensure we have a valid macro first (always at least the key and value)
    if len(tokens) != 4:
        return None

    macro_identifier = tokens[0]

    # Sanity check that the identifier is really one, otherwise error out
    if macro_identifier.kind != TokenKind.IDENTIFIER:
        print_node_tokens(node)
        return None

    # Ensure that we have punctionation in between
    if tokens[2].kind != TokenKind.PUNCTUATION:
        print_node_tokens(node)
        return None

    offset_start = try_parse_c_integer(tokens[1].spelling)
    offset_end = try_parse_c_integer(tokens[3].spelling)

    if offset_start is None or offset_end is None:
        return None

    return NvBitfieldMacroDefinition(
        macro_identifier.spelling, offset_start, offset_end
    )


def try_evaluate_nv_bitfield_qmd_macro(
    node: Cursor,
) -> Optional[NvBitfieldQmdMacroDefinition]:
    assert node.kind == CursorKind.MACRO_DEFINITION

    tokens = list(node.get_tokens())

    is_variable_variant = len(tokens) == 26

    # Ensure we have a valid macro first (always at least the key and value)
    if len(tokens) != 7 and not is_variable_variant:
        return None

    macro_identifier = tokens[0]

    # Sanity check that the identifier is really one, otherwise error out
    if macro_identifier.kind != TokenKind.IDENTIFIER or (
        not is_variable_variant
        and (tokens[1].kind != TokenKind.IDENTIFIER or tokens[1].spelling != "MW")
    ):
        return None

    # Ensure that we have punctionation in between
    if not is_variable_variant and tokens[4].kind != TokenKind.PUNCTUATION:
        return None

    if is_variable_variant:
        offset_end_index = 7
        offset_start_index = 17
        bitfield_next_offset = try_parse_c_integer(tokens[23].spelling)

        if bitfield_next_offset is None:
            return None
    else:
        offset_end_index = 3
        offset_start_index = 5
        bitfield_next_offset = None

    offset_end = try_parse_c_integer(tokens[offset_end_index].spelling)
    offset_start = try_parse_c_integer(tokens[offset_start_index].spelling)

    if offset_start is None or offset_end is None:
        return None

    element_count = 0

    if bitfield_next_offset is not None:
        if "CONSTANT_BUFFER" in macro_identifier.spelling:
            # FIXME: we assume 8 all the time but this can change between archs...
            element_count = 8
        else:
            print("Found variable QMD field but size is unknown")
            return None

        return NvBitfieldQmdMacroDefinition(
            macro_identifier.spelling,
            offset_end,
            offset_start,
            bitfield_next_offset,
            element_count,
        )

    return NvBitfieldQmdMacroDefinition(
        macro_identifier.spelling, offset_end, offset_start
    )


def get_macro_definition_value(
    structs: List[ParsedStruct], constants: List[SimpleMacroDefinition], node: Cursor
) -> Union[
    SimpleMacroDefinition, IoctlMacroDefinition, NvBitfieldMacroDefinition, None
]:
    assert node.kind == CursorKind.MACRO_DEFINITION

    tokens = list(node.get_tokens())

    # Ensure we have a valid macro first (always at least the key and value)
    if len(tokens) < 2:
        return None

    macro_identifier = tokens[0]

    # Sanity check that the identifier is really one, otherwise error out
    if macro_identifier.kind != TokenKind.IDENTIFIER:
        print_node_tokens(node)
        return None

    tokens = tokens[1:]

    res = try_evaluate_simple_macro(constants, node)

    if res is not None:
        return res

    res = try_evaluate_ioctl_macro(structs, constants, node)

    if res is not None:
        return res

    res = try_evaluate_nv_bitfield_macro(node)

    if res is not None:
        return res

    return try_evaluate_nv_bitfield_qmd_macro(node)


def print_nv_bitfield(stream: CodeStream, nv_bitfield: NvBitfieldMacroDefinition):
    func_name = nv_bitfield.name.upper()

    stream.write_line(f"def {func_name}(value: int) -> int:")
    stream.indent()
    stream.write_line(
        f"return set_bits({nv_bitfield.offset_start}, {nv_bitfield.offset_end - nv_bitfield.offset_start}, value)"
    )
    stream.unindent()

    stream.write_line()
    stream.write_line()


def print_nv_qmd(stream: CodeStream, qmd: QmdStruct):
    stream.write_line(f"class {qmd.name}(Structure):")
    stream.indent()

    stream.write_line("_fields_ = [")
    stream.indent()

    unamed_index = 0

    for (field_name, bitfield_end, bitfield_start) in qmd.fields:
        stream.write_line(
            f'("{field_name}", c_uint, {bitfield_end - bitfield_start + 1}),'
        )

    stream.unindent()
    stream.write_line("]")
    stream.unindent()
    stream.write_line()
    stream.write_line()


def print_ioctl_macro(stream: CodeStream, ioctl: IoctlMacroDefinition):
    ioctl_name = ioctl.name.upper()

    arguments = [str(ioctl.type), str(ioctl.nr)]

    ioctl_macro_base = "IO"

    if ioctl.is_write:
        ioctl_macro_base += "W"

    if ioctl.is_read:
        ioctl_macro_base += "R"

    if ioctl.is_write or ioctl.is_read:
        arguments.append(str(ioctl.size))

    stream.write_line(f"{ioctl_name} = {ioctl_macro_base}({', '.join(arguments)})")


def print_ioctl_function(stream: CodeStream, ioctl: IoctlMacroDefinition):
    ioctl_name = ioctl.name.upper()
    func_name = ioctl.name.lower()

    if ioctl.size is not None:
        stream.write_line(f"def {func_name}(fd: Any, arg: {ioctl.size}) -> int:")
        stream.indent()

        stream.write_line(f"return ioctl(fd, {ioctl_name}, pointer(arg))")

        stream.unindent()
    else:
        stream.write_line(f"def {func_name}(fd: Any) -> int:")
        stream.indent()
        stream.write_line(f"return ioctl(fd, {ioctl_name})")
        stream.unindent()

    stream.write_line()
    stream.write_line()


def process_qmd_bitfield(
    qmds: List[QmdStruct], raw_qmd_bitfield: NvBitfieldQmdMacroDefinition
):
    data = raw_qmd_bitfield.name.split("_")
    qmd_name = (data[1] + data[2]).lower()
    field_name = "_".join(data[3:]).lower()

    target_qmd = None

    for qmd in qmds:
        if qmd.name == qmd_name:
            target_qmd = qmd
            break

    if target_qmd is None:
        target_qmd = QmdStruct(qmd_name)

        qmds.append(target_qmd)

    if (
        raw_qmd_bitfield.element_count != 1
        and raw_qmd_bitfield.offset_next_element != 0
    ):
        for i in range(raw_qmd_bitfield.element_count):
            special_field_name = f"{field_name}_{i}"
            offset_end = (
                raw_qmd_bitfield.offset_end + i * raw_qmd_bitfield.offset_next_element
            )
            offset_start = (
                raw_qmd_bitfield.offset_start + i * raw_qmd_bitfield.offset_next_element
            )
            target_qmd.fields.append((special_field_name, offset_end, offset_start))
    else:
        target_qmd.fields.append(
            (field_name, raw_qmd_bitfield.offset_end, raw_qmd_bitfield.offset_start)
        )


def parse_header(target_header: str, output_file_path: str, node: Cursor) -> None:
    structs: List[ParsedStruct] = list()
    constants: List[SimpleMacroDefinition] = list()
    ioctls: List[IoctlMacroDefinition] = list()
    nv_bitfields: List[NvBitfieldMacroDefinition] = list()
    raw_qmd_bitfields: List[NvBitfieldQmdMacroDefinition] = list()

    possible_forward_declaration_macros: List[Cursor] = list()

    childs: Iterable[Cursor] = filter_uneeded_stuffs(target_header, node)

    for child in childs:
        if child.kind == CursorKind.STRUCT_DECL:
            obj = create_struct(child)

            if obj is not None:
                structs.append(obj)

        elif child.kind == CursorKind.UNION_DECL:
            # TODO
            raise Exception("TODO")
        elif child.kind == CursorKind.MACRO_DEFINITION:
            macro_def = get_macro_definition_value(structs, constants, child)
            if macro_def is not None:
                if type(macro_def) is SimpleMacroDefinition:
                    constants.append(macro_def)
                elif type(macro_def) is IoctlMacroDefinition:
                    ioctls.append(macro_def)
                elif type(macro_def) is NvBitfieldMacroDefinition:
                    nv_bitfields.append(macro_def)
                elif type(macro_def) is NvBitfieldQmdMacroDefinition:
                    raw_qmd_bitfields.append(macro_def)
                else:
                    raise Exception("TODO")
            else:
                possible_forward_declaration_macros.append(child)
        elif child.kind == CursorKind.MACRO_INSTANTIATION:
            pass
        else:
            print_node(child)
            print_recursive_childs(child)
            print_arguments(child)

    # Try to resolve possible forward declared macros
    for child in possible_forward_declaration_macros:
        macro_def = get_macro_definition_value(structs, constants, child)

        if macro_def is not None:
            if type(macro_def) is SimpleMacroDefinition:
                constants.append(macro_def)
            elif type(macro_def) is IoctlMacroDefinition:
                ioctls.append(macro_def)
            # Shouldn't be possible
            elif type(macro_def) is NvBitfieldMacroDefinition:
                nv_bitfields.append(macro_def)
            # Shouldn't be possible
            elif type(macro_def) is NvBitfieldQmdMacroDefinition:
                raw_qmd_bitfields.append(macro_def)
            else:
                raise Exception("TODO")
        else:
            # print_node_tokens(child)
            pass

    # Process QMD definitions
    qmds: List[QmdStruct] = list()

    for raw_qmd_bitfield in raw_qmd_bitfields:
        process_qmd_bitfield(qmds, raw_qmd_bitfield)

    for qmd in qmds:
        # Ensure the bitfield is sorted correctly
        qmd.fields.sort(key=lambda data: data[2])

    # Now let's output
    stream = create_codestream(datetime.now())

    if len(structs) != 0 or len(qmds) != 0:
        stream.write_line("from ctypes import *")

    if len(ioctls) != 0 or len(nv_bitfields) != 0:
        stream.write_line("from utils import *")

    stream.write_line()
    stream.write_line()

    if len(ioctls) != 0:
        stream.write_line("libc = CDLL('libc.so.6')")
        stream.write_line("ioctl = libc.ioctl")

    for constant in constants:
        assert type(constant.value) is int
        stream.write_line(
            f"{constant.name}: {type(constant.value).__name__} = 0x{constant.value:X}"
        )

    stream.write_line()
    stream.write_line()

    for nv_bitfield in nv_bitfields:
        print_nv_bitfield(stream, nv_bitfield)

    stream.write_line()
    stream.write_line()

    for qmd in qmds:
        print_nv_qmd(stream, qmd)

    stream.write_line()
    stream.write_line()

    for struct in structs:
        print_struct(stream, struct)

    for ioctl in ioctls:
        print_ioctl_macro(stream, ioctl)

    stream.write_line()
    stream.write_line()

    for ioctl in ioctls:
        print_ioctl_function(stream, ioctl)

    with open(output_file_path, "w+") as result_file:
        result_file.write(stream.get())
