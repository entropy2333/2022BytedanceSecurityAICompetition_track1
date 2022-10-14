## oov emoji

import pandas as pd
import copy
import json
import emojiswitch  # emoji转换库（0.52 粗看效果一般）
import re
import difflib  # 文本diff检测库
from collections import defaultdict, Counter
import emoji
emoji.demojize("😊",language="en")
# import paddlehub as hub
# module = hub.Module(name='baidu_translate')


import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element
unicode_map = dict()
tree = ET.parse("./data/unicode-org.xml")
root = tree.getroot()
for child in root:
    if child.tag == 'annotations':
        for anno in child[698:3811]:
            e = anno.attrib['cp'] 
            # if 'type' in anno.attrib and e in unicode_map:
            #     continue
            ts = anno.text.strip().split("|")
            idx = 0
            t = ""
            while idx < len(ts):
                # 0922 add start
                t = ts[idx]
                if t in ['人', "男", "女"]:
                    idx += 1
                    continue
                if '的猫' in t:
                    idx += 1
                    continue
                if '禁止' not in t and len(t) >= 5:
                    idx += 1
                    continue
                if '的男人' in t or '的女人' in t:
                    t = t[:t.index('的')]
                if '的' in t:
                    t = t[t.index('的')+1:]
                if '者' in t:
                    t = t.replace("者", '')
                if '女生' in t or "男生" in t or '手势' in t or '按钮' in t or '箭头' in t:
                    t = t.replace("女生", "").replace("男生", "").replace("手势", "").replace("按钮", "").replace("箭头", "")
                # 0922 add end
                
                break
            if idx < len(ts):
                unicode_map[e] = t


train = pd.read_excel("./data/emoji7w.xlsx")
test = pd.read_csv("./data/emoji7w-test_data.csv", sep='\t')

emojis = emoji.distinct_emoji_list(''.join(train['原emoji文本（不可编辑）'].to_list()+test['prediction'].to_list()))

""""""
# 构造map，选择出现最多的text
diffInstance = difflib.Differ()
emoji2texts = defaultdict(list)
for idx, pred in enumerate(train['纯文本（可编辑）']):
    modi = train['修改后的emoji文本（可编辑）'][idx]
    diffList = list(diffInstance.compare(modi, pred))
    emo = ''
    text = ''
    for s in diffList:
        if s[0] == '-':
            emo += s[-1]
        elif s[0] == '+':
            text += s[-1]
        elif emo != '' and text != '':
            emoji2texts[emo.replace(" ", "")].append(text)  # v3【0916 14:37】add replace 消除空格
            emo = ''
            text = ''
# 只添加出现最多的text
multi = dict()
emoji2text = dict()
for e, t in emoji2texts.items():
    _ = Counter(t)
    # print(_)
    if len(_) > 1:
        l = [k for k, c in list(_.items())[:3] if c > 2]
        if len(l):
            multi[e] = l
    text, freq = _.most_common(1)[0]
    """0922 添加 频率大于2"""
    if freq >= 2:
        emoji2text[e] = text
print("origin_map_len:{}, emoji2text_len:{}, multi_cnt:{}".format(len(emoji2texts), len(emoji2text), len(multi)))
emoji2text = dict(sorted(list(emoji2text.items()),key=lambda x: -len(x[0])))
with open("./data/map_multi.json", "w") as f:
    json.dump(multi,f,ensure_ascii=False, indent=4)
with open("./data/map_mostCommon.json", "w") as f:
    json.dump(emoji2text,f,ensure_ascii=False, indent=4)

# 
## 补充一些公开数据的map
new_emoji2text = dict()

# emoji2text中过长的key
long_ks = list()
for k,v in emoji2text.items():
    if len(emoji.distinct_emoji_list(k))==0 and re.match("[\u4E00-\u9FA5a-zA-Z0-9]", k) and len(k) == 1:
        # print(k,v)
        continue
    if re.match("[\u4E00-\u9FA5]", k[0]) and k[1:] in emoji2text:   # v4 0917 1224 添加，去除 金💍->戒指 这种
        continue
    vs = v.split(" ")
    if k not in new_emoji2text:
        new_emoji2text[k] = v
    if len(emoji.distinct_emoji_list(k)) > 1 and len(vs) > 2:
        long_ks.extend(list(k))
for emo in long_ks:
    if emo in unicode_map:
        if emo not in new_emoji2text:
            new_emoji2text[emo] = unicode_map[emo]

new_emoji2text = dict(sorted(list(new_emoji2text.items()), key=lambda x: -len(x[0])))
print(f"long_ks:{len(long_ks)}, new_emoji2text:{len(new_emoji2text)}")
json.dump(  new_emoji2text,
            open("./data/map_v5.json", "w"),
            ensure_ascii=False, indent=4)
""""""



# ------------------
# oov

oov_map = dict()
skins = [ 
    '_dark_skin_tone', 
    '_medium_skin_tone', 
    '_light_skin_tone', 
    '_medium-dark_skin_tone', 
    '_medium-light_skin_tone'
]
# # 0917_v4
# for e in emojis:
#     if e not in new_emoji2text:
#         en = emoji.demojize(e)
#         # if "button" in en:
#         #     print(e,en)
#         if 'skin' in en:
#             for s in skins:
#                 en = en.replace(s, "")
#             emo = emoji.emojize(en)
#             if emo in new_emoji2text:
#                 oov_map[e] = new_emoji2text[emo]
#                 # print(e, new_emoji2text[emo])
                
#         # elif 'keycap' in en:
#         #     oov_map[e] = en.replace("keycap_", "")
#             # print(e,en.replace("keycap_", ""))
#         # else:
#         #     en = en.split("with")[0].replace("_"," ").replace(":", "").replace("sign","")
#         #     ch = module.translate(en, 'en', 'zh')
#         #     ch = ch.split("的")[0]
#         #     if '在' in ch:
#         #         ch = ch[-2:]
#         #     # if len(ch) <= 4:
#         #     oov_map[e] = ch
#         #     print(e, ch)
# ####

for e, t in unicode_map.items():
    if e not in oov_map.items():
        oov_map[e] = unicode_map[e]
# for e, t in oov_map.items():
#     if t in [f":{i}:" for i in range(10)]:
#         oov_map[e] = t[1]

special = {
    "‼": "吃惊",
    "‼": "吃惊",
    "⁉": "吃惊",
    "⁉": "吃惊",
    "❓": "疑惑",
    "❓": "疑惑",
    "❔": "疑惑",
    "❔": "疑惑",
    "❕": "吃惊",
    "❕": "吃惊",
    "❗": "吃惊",
    "❗": "吃惊",
}

for e,t in special.items():
    oov_map[e] = t

for e, t in oov_map.items():
    if t in [f":{i}:" for i in range(10)]:
        oov_map[e] = t[1]
oov_map = dict(sorted(list(oov_map.items()),key=lambda x: -len(x[0])))
json.dump(oov_map, open("./data/oov.json", "w"), ensure_ascii=False, indent=4)

