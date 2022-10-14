import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from emoji import is_emoji

sys.path.append(str(Path(__file__).parent.parent.parent))
from icecream import ic
from tqdm import tqdm
from utils import write2json


def parse_xml(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    return root


def process_text(t):
    if t in ['人', "男", "女"]:
        return t
    if '的猫' in t:
        return t
    # if '禁止' in t:
    #     t = "禁止"
    if '的男人' in t or '的女人' in t:
        t = t[:t.index('的')]
    if '的' in t:
        t = t[t.index('的') + 1:]
    if '者' in t:
        t = t.replace("者", '')
    if '女生' in t or "男生" in t or '手势' in t or '按钮' in t:
        t = t.replace("女生", "").replace("男生", "").replace("手势", "").replace("按钮", "")
    if "肤色" in t:
        t = t.replace("较浅肤色", "") \
            .replace("较深肤色", "") \
            .replace("中等-浅肤色", "") \
            .replace("中等肤色", "") \
            .replace("中等-深肤色", "") \
            .replace("肤色", "")
    if "成人成人" in t:
        t = t.replace("成人成人", "")
    if "旗: " in t:
        t = t.replace("旗: ", "")
    # if t == "旗":
    #     t = ""
    t = t.strip().rstrip("较").rstrip(":").rstrip("：")
    return t


def get_emoji_cldr_annotations(root):
    annotations = root.findall("annotations/annotation")
    print(f"Found {len(annotations)} annotations")
    emoji2text = {}
    emoji2text_tts = {}
    for annotation in tqdm(annotations):
        char = annotation.attrib["cp"]
        if is_emoji(char):
            if annotation.attrib.get("type") == "tts":
                tts_text = process_text(annotation.text)
                if tts_text:
                    emoji2text_tts[char] = tts_text
            else:
                text_list = annotation.text.replace(" ", "").split("|")
                text_list = [process_text(x) for x in text_list]
                text_list = [x for x in text_list if x]
                if text_list:
                    emoji2text[char] = list(set([x for x in text_list if x]))
    # assert len(emoji2text) == len(emoji2text_tts)
    print(f"Found {len(emoji2text)} pair emoji annotations")
    return emoji2text, emoji2text_tts


str1 = "有络腮胡子的男人"
ic(process_text(str1), str1[:str1.index("的")])

root = parse_xml("data/cldr_annotations_zh.xml")
emoji2text, emoji2text_tts = get_emoji_cldr_annotations(root)

root = parse_xml("data/cldr_annotations_derived_zh.xml")
emoji2text_derived, emoji2text_tts_derived = get_emoji_cldr_annotations(root)

emoji2text.update(emoji2text_derived)
emoji2text_tts.update(emoji2text_tts_derived)
write2json(emoji2text, "data/emoji2text_cldr.json", ensure_ascii=False)
write2json(emoji2text_tts, "data/emoji2text_cldr_tts.json", ensure_ascii=False)
