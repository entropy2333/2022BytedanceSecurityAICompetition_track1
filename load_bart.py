import argparse
import random
from pathlib import Path

import pandas as pd
from icecream import ic
from utils import load_json
from tqdm import tqdm

from constants import (CSV_SEP, DF_TEST_RAW_PATH, EMOJI2ZH_PATH, GENERATION_KWARGS)
from data_process import cldr_demojize, process_text
from simplet5 import SimpleT5


def predict_main(args):
    df_test = pd.read_csv(DF_TEST_RAW_PATH, sep=CSV_SEP)

    #FIXME: optional
    emoji2zh = load_json(EMOJI2ZH_PATH)
    # emoji2zh.update(load_json("data/emoji2text_unique.json"))
    df_test["prediction"] = df_test["prediction"].apply(process_text)
    df_test['prediction'] = df_test['prediction'].apply(lambda x: cldr_demojize(x, emoji2zh))
    df_test['prediction'] = df_test['prediction'].apply(lambda x: x.replace(" ", ""))

    ic(df_test.head())
    ic(df_test.tail())

    model = SimpleT5()
    # let's load the trained model for inferencing:
    model.load_model("bart", args.checkpoint, use_gpu=True)

    generation_kwargs = GENERATION_KWARGS
    ic(generation_kwargs)

    batch_size = args.batch_size
    for idx in tqdm(range(0, df_test.shape[0], batch_size)):
        raw_texts = df_test['prediction'][idx:idx + batch_size]
        raw_texts = list(raw_texts.values)
        preds = model.batch_predict(raw_texts, batch_size=batch_size, **generation_kwargs)
        for i, pred in enumerate(preds):
            df_test.loc[idx + i, 'prediction'] = pred.replace(" ", "")
        if idx < 10 or random.random() < 0.1:
            ic(idx, raw_texts[0], preds[0])
    df_test.to_csv(args.submit_file, sep=CSV_SEP, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",
                        "--ckpt",
                        type=str,
                        default="outputs/simplet5-epoch-9-train-loss-0.0089-val-loss-0.0043")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--submit_file", type=str, default="submit/0926_bart_beam_large_pseudo.csv")
    args = parser.parse_args()
    ic(args)
    predict_main(args)
