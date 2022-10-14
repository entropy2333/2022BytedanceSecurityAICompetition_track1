import json
import re
import string
from functools import partial
from itertools import chain
from typing import Dict, List

FULL_WIDTH_CHARACTERS_A_Z = list(range(0xFF21, 0xFF3B))
FULL_WIDTH_CHARACTERS_a_z = list(range(0xFF41, 0xFF5B))
FULL_WIDTH_CHARACTERS_0_9 = list(range(0xFF10, 0xFF1A))
HALF_WIDTH_CHARACTERS_A_Z = list(range(0x0041, 0x005B))
HALF_WIDTH_CHARACTERS_a_z = list(range(0x0061, 0x007B))
HALF_WIDTH_CHARACTERS_0_9 = list(range(0x0030, 0x003A))
FULL_WIDTH_CHARACTERS_TOTAL = list(range(0xFF01, 0xFF5E + 1))
HALF_WIDTH_CHARACTERS_TOTAL = list(range(0x0021, 0x007E + 1))
FULL_WIDTH_CHARACTERS_PUNCTUATION = list(range(0xFF01, 0xFF0F + 1)) + \
                                    list(range(0xFF1A, 0xFF20 + 1)) + \
                                    list(range(0xFF3B, 0xFF40 + 1)) + \
                                    list(range(0xFF5B, 0xFF5E + 1))
HALF_WIDTH_CHARACTERS_PUNCTUATION = list(range(0x0021, 0x002F + 1)) + \
                                    list(range(0x003A, 0x0040 + 1)) + \
                                    list(range(0x005B, 0x0060 + 1)) + \
                                    list(range(0x007B, 0x007E + 1))


def get_full_or_width_characters(mode='full',
                                 include_a_z=True,
                                 include_A_Z=True,
                                 include_0_9=True,
                                 include_punctuation=True,
                                 index=False):
    index_list = []
    if include_a_z:
        index_list.extend(FULL_WIDTH_CHARACTERS_a_z if mode == 'full' else HALF_WIDTH_CHARACTERS_a_z)
    if include_A_Z:
        index_list.extend(FULL_WIDTH_CHARACTERS_A_Z if mode == 'full' else HALF_WIDTH_CHARACTERS_A_Z)
    if include_0_9:
        index_list.extend(FULL_WIDTH_CHARACTERS_0_9 if mode == 'full' else HALF_WIDTH_CHARACTERS_0_9)
    if include_punctuation:
        index_list.extend(FULL_WIDTH_CHARACTERS_PUNCTUATION if mode == 'full' else HALF_WIDTH_CHARACTERS_PUNCTUATION)
    if index:
        return index_list
    return [chr(i) for i in index_list]


get_full_width_characters = partial(get_full_or_width_characters, mode='full')
get_half_width_characters = partial(get_full_or_width_characters, mode='half')


def convert_full_and_half_width(
    text,
    mode="full2half",
    include_a_z=True,
    include_A_Z=True,
    include_0_9=True,
    include_punctuation=True,
):
    """
    Convert full-width characters to half-width.
    """
    full_width_characters = get_full_or_width_characters(mode='full',
                                                         include_a_z=include_a_z,
                                                         include_A_Z=include_A_Z,
                                                         include_0_9=include_0_9,
                                                         include_punctuation=include_punctuation,
                                                         index=True)
    half_width_characters = get_full_or_width_characters(mode='half',
                                                         include_a_z=include_a_z,
                                                         include_A_Z=include_A_Z,
                                                         include_0_9=include_0_9,
                                                         include_punctuation=include_punctuation,
                                                         index=True)
    assert len(full_width_characters) == len(half_width_characters), (len(full_width_characters), len(half_width_characters))
    if mode == "half2full":
        full_half_map = dict(zip(half_width_characters, full_width_characters))
    elif mode == "full2half":
        full_half_map = dict(zip(full_width_characters, half_width_characters))
    else:
        raise ValueError("mode must be 'half2full' or 'full2half'")
    return text.translate(full_half_map)


convert_full2half_width = partial(convert_full_and_half_width, mode='full2half')
convert_half2full_width = partial(convert_full_and_half_width, mode='half2full')


def load_json(json_file):
    with open(json_file, "r", encoding="utf8") as fin:
        data = json.load(fin)
    print(f"load {len(data)} from {json_file}")
    return data


def write2json(data, data_path, data_name="data", ensure_ascii=False, indent=2):
    with open(data_path, "w", encoding="utf-8") as fout:
        fout.write(json.dumps(data, ensure_ascii=ensure_ascii, indent=indent))
    print(f"{data_name}({len(data)}) saved into {data_path}")
