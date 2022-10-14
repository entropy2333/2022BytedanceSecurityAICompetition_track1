import pandas as pd
import copy
import json
import emojiswitch  # emoji转换库（0.52 粗看效果一般）
import re
import difflib  # 文本diff检测库
from emoji2text import emoji2text as e2t_en
from collections import defaultdict, Counter
import emoji
emoji.demojize("😊",language="en")

test = pd.read_csv("./data/emoji7w-test_data.csv", sep='\t')


# 0.82
not_good = 0
having_multi_cnt = 0
res = copy.deepcopy(test)
multi = json.load(open("./data/map_multi.json","r"))
new_emoji2text = json.load(open("./data/map_v5.json","r"))
oov = json.load(open("./data/oov.json", "r"))
for e,t in oov.items():
    if e not in new_emoji2text:
        new_emoji2text[e] = t
new_emoji2text = dict(sorted(list(new_emoji2text.items()),key=lambda x: -len(x[0])))

for idx, raw in enumerate(res['prediction']):
    pred = raw
    # for i in range(5):
    #     pred = pred.replace(chr(127995+i),'')
    for e in multi.keys():
        if e in pred:
            having_multi_cnt += 1
            # print(e, pred)
            break
            
    for e, t in new_emoji2text.items():
        pred = pred.replace(e, t)

    res['prediction'][idx] = pred
    
print(not_good, having_multi_cnt)
res.to_csv("./submit/test_map_v5.csv", sep="\t", index=False)