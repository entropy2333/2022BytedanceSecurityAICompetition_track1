import argparse
from pathlib import Path

import pandas as pd
from icecream import ic
from sklearn.model_selection import train_test_split

from constants import CSV_SEP
from simplet5 import SimpleT5


def train_main(args):
    df_train_path = args.df_train
    ic(df_train_path)
    df_total = pd.read_csv(df_train_path, sep=CSV_SEP)

    if args.df_valid:
        df_valid = pd.read_csv(args.df_valid, sep=CSV_SEP)
        df_train = df_total
    else:
        df_train, df_valid = train_test_split(df_total, test_size=0.1, random_state=42)

    if args.train_with_total:
        df_train = df_total

    ic(df_train.shape, df_valid.shape)

    ic(df_train.head())
    ic(df_valid.head())

    ic(df_train["source_text"].str.len().describe())
    ic(df_train["target_text"].str.len().describe())
    ic(df_valid["source_text"].str.len().describe())
    ic(df_valid["target_text"].str.len().describe())
    ic(df_train[df_train["source_text"] == df_train["target_text"]].shape)
    ic(df_train[df_train["source_text"] != df_train["target_text"]].shape)

    # df_train = df_train[df_train["source_text"] != df_train["target_text"]]
    # df_train.reset_index(drop=True, inplace=True)

    emoji_dict = {}
    ic(len(emoji_dict))
    special_tokens = list(emoji_dict.keys())

    # %%
    model = SimpleT5()

    pretrain_path = args.pretrain_path
    ic(pretrain_path)

    model.from_pretrained(
        model_type=args.model_type,
        model_name=pretrain_path,
        additional_special_tokens=special_tokens,
    )

    ic(len(model.tokenizer))
    ic(model.tokenizer.special_tokens_map)

    model.train(
        train_df=df_train,
        eval_df=df_valid,
        source_max_token_len=args.source_max_token_len,
        target_max_token_len=args.target_max_token_len,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        use_gpu=args.use_gpu,
        output_dir=args.output_dir,
        early_stopping_patience_epochs=args.early_stopping_patience_epochs,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        learning_rate=args.learning_rate,
        dataloader_num_workers=args.dataloader_num_workers,
        use_fgm=args.use_fgm,
        gradient_clip_algorithm=args.gradient_clip_algorithm,
        gradient_clip_val=args.gradient_clip_val,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--df_train", type=str, default="data/df_total_demojized_pseudo.csv")
    parser.add_argument("--df_valid", type=str, default="")
    parser.add_argument("--train_with_total", type=bool, default=True)
    parser.add_argument("--model_type", type=str, default="bart")
    parser.add_argument("--pretrain_path", type=str, default="fnlp/bart-large-chinese")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--early_stopping_patience_epochs", type=int, default=0)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--use_fgm", type=bool, default=True)
    parser.add_argument("--gradient_clip_algorithm", type=str, default=None)
    parser.add_argument("--gradient_clip_val", type=float, default=None)
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--source_max_token_len", type=int, default=50)
    parser.add_argument("--target_max_token_len", type=int, default=50)
    args = parser.parse_args()
    ic(args)
    train_main(args)
