import pandas as pd
import argparse
punc_map = {
    ",": "，",
    "...": "…",
    "!": "！",
    "?": "？",
    "(": "（",
    ")": "）",
    "#": "＃",
    ".": "。",
    ":": "：",
    ";": "；"
}
    

def ban2quan(file):
    raw_test = pd.read_csv("./data/emoji7w-test_data.csv", sep="\t")
    data = pd.read_csv(file, sep="\t")
    for i, raw in enumerate(raw_test['prediction']):
        pred = data['prediction'][i]
        for ban, quan in punc_map.items():
            if quan in raw and ban in pred:
                pred = pred.replace(ban, quan)
        data['prediction'][i] = pred.replace(" ","")
    data.to_csv(file.replace(".csv","_h2f.csv"), sep="\t",index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, help="file")

    args = parser.parse_args()

    ban2quan(args.file)