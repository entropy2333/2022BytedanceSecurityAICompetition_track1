import argparse

import emoji
import pandas as pd
# from emoji import demojize
from emojiswitch import demojize
from icecream import ic
from tqdm import tqdm

from constants import CSV_SEP, DF_TEST_RAW_PATH, EMOJI2ZH_PATH
from data_process import cldr_demojize, process_text
from utils import load_json


def main(args):
    df_test = pd.read_csv(DF_TEST_RAW_PATH, sep=CSV_SEP)
    ic(df_test.shape)

    df_pseudo = pd.read_csv(args.pseudo_path, sep=CSV_SEP)
    ic(df_pseudo.shape)
    # df_ge5 = pd.read_csv("data/pesudo_ge5_res84409_res84190_res84091_res83890_res83692.csv", sep="\t")
    # ic(df_ge5.shape)

    df_pseudo["source_text"] = ""
    for idx, row in tqdm(df_pseudo.iterrows()):
        id = row["id"]
        prediction = row["prediction"]
        source_text = df_test[df_test["id"] == id]["prediction"]
        source_text = str(source_text.values[0])
        if idx < 10:
            ic(id, str(prediction), source_text)
        df_pseudo["source_text"][idx] = source_text

    df_pseudo = df_pseudo[["source_text", "prediction"]]
    df_pseudo.columns = ["source_text", "target_text"]

    # df_pseudo.to_csv("data/df_pseudo.csv", index=False)

    df_pseudo['target_text'] = df_pseudo['target_text'].apply(process_text)
    df_pseudo['target_text'] = df_pseudo['target_text'].apply(lambda x: x.replace(" ", ""))
    # df_anno_subset[['source_text', 'target_text']].to_csv("data/df_anno.csv", index=False, sep="\t")

    emoji2zh = load_json(EMOJI2ZH_PATH)
    df_pseudo['source_text'] = df_pseudo['source_text'].apply(process_text)
    df_pseudo['source_text'] = df_pseudo['source_text'].apply(lambda x: cldr_demojize(x, emoji2zh))
    df_pseudo['source_text'] = df_pseudo['source_text'].apply(lambda x: x.replace(" ", ""))

    df_train = pd.read_csv("data/df_total_demojized.csv", sep=CSV_SEP)

    df_output = pd.concat([df_train, df_pseudo], axis=0)

    ic(df_train.shape, df_output.shape)

    df_output.to_csv("data/df_total_demojized_pseudo.csv", sep=CSV_SEP, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pseudo_path", "-f", type=str, default="data/df_pseudo.csv")
    args = parser.parse_args()
    main(args)
