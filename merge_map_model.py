
import difflib
import pandas as pd
import copy
import re
import json
import argparse
from tqdm import tqdm
import emoji
diffInstance = difflib.Differ()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test", type=str, 
                default="./data/emoji7w-test_data.csv", help="raw test file")
    parser.add_argument("-m", "--map", type=str, help="source map file")
    parser.add_argument("-p", "--model", type=str, help="model output file")
    parser.add_argument("-o", "--output", type=str, help="output file")

    args = parser.parse_args()
    multi = json.load(open("./data/map_multi.json", "r"))
    emoji_map = json.load(open("./data/map_v5.json","r"))
    oov = json.load(open("./data/oov.json","r"))
    keys = [emo for emo in list(emoji_map.keys()) + list(oov.keys()) if len(emoji.distinct_emoji_list(emo)) > 0 and emoji.distinct_emoji_list(emo)[0]==emo]

    raw_test = pd.read_csv(args.test, sep="\t")
    map_0916 = pd.read_csv(args.map, sep="\t")
    model_en = pd.read_csv(args.model, sep="\t")
    res = copy.deepcopy(model_en)
    same = 0
    oov_cnt = 0
    cnt = 0
    too_long = 0

    for i, raw in enumerate(raw_test['prediction']):
        map_pred = map_0916['prediction'][i]
        model_pred = model_en['prediction'][i]
        if map_pred == model_pred:
            same += 1
            continue
        
        raw_noemoji = raw.replace(" ","")
        use_map = False
           
        for e in keys:
            if e in raw_noemoji:
                raw_noemoji = raw_noemoji.replace(e, "")

                
        
        # v1
        raw_noemoji = re.sub("([^\u4E00-\u9FA5a-zA-Z0-9]+)","",raw_noemoji)
        model_noemoji = re.sub("([^\u4E00-\u9FA5a-zA-Z0-9]+)","",model_pred)
       
        if len(model_noemoji) > 3 * max(1,len(emoji.distinct_emoji_list(raw))) + len(raw_noemoji):
            too_long += 1
        difflist = list(diffInstance.compare(raw_noemoji, model_noemoji))
        differ_str = ''        
        for d in difflist:
            if d[0] == '-':
                if d[-1] in 'QWERTYUIOPASDFGHJKLZXCVBNMqwertyuiopasdfghjklzxcvbnm':
                    differ_str += d[-1]
                else:
                    use_map = True
                    break
            if d[0] == '+' and differ_str != '':
                if d[-1].upper() == differ_str[0].upper():
                    model_pred = model_pred.replace(d[-1], differ_str[0])
                    differ_str = differ_str[1:]
        res['prediction'][i] = model_pred.replace("人工驾驶","跑步")
        if use_map:
            cnt += 1
            print(raw, model_pred, map_pred)
            res['prediction'][i] = map_pred.replace("人工驾驶","跑步")
    print(f"same:{same}, use_map:{cnt}, too_long:{too_long}")#oov_cnt:{oov_cnt}
    res.to_csv(args.output, sep="\t", index=False)