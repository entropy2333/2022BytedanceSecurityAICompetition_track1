import argparse
import json
from functools import partial
from pathlib import Path

import emoji
import emojiswitch
import pandas as pd
from icecream import ic
from tqdm import tqdm, trange

from constants import (BOS_TOKEN, CSV_SEP, DF_TEST_RAW_PATH, DF_TRAIN_RAW_PATH,
                       EMOJI2ZH_PATH, EMOJI_ZH, EOS_TOKEN, TASK_PREFIX)
from utils import convert_full2half_width, load_json, write2json

CUSTOM_EMOJI_MAP = {u"🐻\u200d❄️": "北极", u"🤵\u200d♂️": "帅哥", u"👰\u200d♀️": "婚纱", u"🤵\u200d♀️": "男"}

full2half = lambda x: convert_full2half_width(x, include_a_z=False, include_A_Z=False, include_0_9=False)


def process_text(text: str, do_full2half=True):
    text = text.lower()
    if do_full2half:
        text = full2half(text)
    # text = text.replace("\u200d", "")
    # text = text.replace("♀️", "女性")
    # text = text.replace("♂️", "男性")
    return text


def load_train_data(prefix=TASK_PREFIX, do_filter=True, do_full2half=True):
    df_train = pd.read_excel(DF_TRAIN_RAW_PATH)

    ic(df_train.shape)
    df_train.head()

    df_train = df_train.rename(columns={
        "原emoji文本（不可编辑）": "raw_text",
        "修改后的emoji文本（可编辑）": "source_text",
        "纯文本（可编辑）": "target_text",
    })
    # FIXME
    df_train = df_train[["source_text", "target_text"]]
    df_train.head()
    df_train["source_text"] = prefix + df_train["source_text"]
    for col in ['source_text', 'target_text']:
        df_train[col] = df_train[col].apply(lambda x: process_text(x, do_full2half=do_full2half))
    df_train['target_text'] = df_train['target_text'].apply(lambda x: x.replace(" ", ""))
    # 过滤掉没有emoji的
    if do_filter:
        from emoji import emoji_count
        df_train = df_train[df_train["source_text"] != df_train["target_text"]]
        df_train['emoji_count'] = df_train['source_text'].apply(lambda x: emoji_count(x))
        df_train = df_train[df_train['emoji_count'] > 0]
        df_train.reset_index(drop=True, inplace=True)
    df_train = df_train[["source_text", "target_text"]]
    return df_train


def load_test_data():
    df_test = pd.read_csv(DF_TEST_RAW_PATH, sep=CSV_SEP)
    return df_test


def build_emoji_vocab(df_train, df_test=None):

    emoji_vocab = {}
    emoji_counter = {}
    emoji_vocab_test = {}
    emoji_counter_test = {}
    total_emoji_count = 0
    for idx, text in tqdm(enumerate(df_train["source_text"])):
        emojis = [x['emoji'] for x in emoji.emoji_list(text)]
        for em in emojis:
            total_emoji_count += 1
            emoji_counter[em] = emoji_counter.get(em, 0) + 1
            emoji_vocab[em] = emoji_vocab.get(em, len(emoji_vocab))
    ic(total_emoji_count)
    if df_test is not None:
        for idx, text in tqdm(enumerate(df_test["prediction"])):
            emojis = [x['emoji'] for x in emoji.emoji_list(text)]
            for em in emojis:
                total_emoji_count += 1
                emoji_counter_test[em] = emoji_counter_test.get(em, 0) + 1
                emoji_vocab_test[em] = emoji_vocab_test.get(em, len(emoji_vocab_test))
    ic(total_emoji_count)
    print(f"emoji_vocab: {len(emoji_vocab)} emoji_vocab_test: {len(emoji_vocab_test)}")
    print(f"emoji_train: {sum(emoji_counter.values())} emoji_test: {sum(emoji_counter_test.values())}")
    write2json(emoji_vocab, "data/emoji_vocab.json")
    emoji_counter = dict(sorted(emoji_counter.items(), key=lambda x: x[1], reverse=True))
    write2json(emoji_counter, "data/emoji_counter.json")
    write2json(emoji_vocab_test, "data/emoji_vocab_test.json")
    emoji_counter_test = dict(sorted(emoji_counter_test.items(), key=lambda x: x[1], reverse=True))
    write2json(emoji_counter_test, "data/emoji_counter_test.json")


def build_emoji_map(df_train):
    import difflib  # 文本diff检测库

    from emoji import is_emoji
    diff = difflib.Differ()
    emoji2text = dict()

    # version1
    for idx, target in enumerate(df_train['target_text']):
        source = df_train['source_text'][idx]
        diff_list = list(diff.compare(source, target))
        emoji = ''
        text = ''
        for s in diff_list:
            if s[0] == '-':
                emoji += s[-1]
            elif s[0] == '+':
                text += s[-1]
            elif emoji != '' and text != '':
                emoji2text[emoji] = emoji2text.get(emoji, []) + [text]
                emoji2text[emoji] = list(set(emoji2text[emoji]))
                emoji = ''
                text = ''

    # num_samples = len(df_train)
    # for idx in trange(num_samples):
    #     source_text = df_train.loc[idx, "source_text"]
    #     target_text = df_train.loc[idx, "target_text"]
    #     diff_seq = difflib.SequenceMatcher(lambda x: x == " ", source_text, target_text)
    #     for opcode, i1, i2, j1, j2 in diff_seq.get_opcodes():
    #         if opcode == 'equal':
    #             continue
    #         elif opcode == 'replace':
    #             if is_emoji(source_text[i1:i2]) and not is_emoji(target_text[j1:j2]):
    #                 emoji2text[source_text[i1:i2]] = target_text[j1:j2]
    #         else:
    #             continue
    #             print(f"{idx:5} {opcode:7} {source_text[i1:i2]} -> {target_text[j1:j2]}")
    # print(f"{idx:5} {source_text} -> {target_text}")

    write2json(emoji2text, "data/emoji2text.json")
    emoji2text_unique = {}
    for em, text_list in emoji2text.items():
        if is_emoji(em) and len(text_list) == 1:
            emoji2text_unique[em] = text_list[0]
    ic(len(emoji2text_unique))
    write2json(emoji2text_unique, "data/emoji2text_unique.json")
    return emoji2text, emoji2text_unique


def filter_unique_map(df_train, emoji2text_unique):
    ic(df_train.shape)
    diff_count = 0
    for idx, source in tqdm(enumerate(df_train['source_text'])):
        source_to_replace = source
        target_text = df_train['target_text'][idx]
        find_flag = False
        for em, text in emoji2text_unique.items():
            if em in source_to_replace:
                find_flag = True
                source_to_replace = source_to_replace.replace(em, text)
        if not find_flag:
            continue
        if source_to_replace != target_text:
            ic(idx, source, target_text, source_to_replace)
            diff_count += 1
        df_train['source_text'][idx] = source_to_replace
    ic(diff_count)
    df_train = df_train[df_train['source_text'] != df_train['target_text']]
    df_train.reset_index(drop=True, inplace=True)
    ic(df_train.shape)
    return df_train


def cldr_demojize(raw_text, emoji2zh, return_emojis=False):
    emojis = [x['emoji'] for x in emoji.emoji_list(raw_text)]
    text = raw_text
    for em in emojis:
        if em not in emoji2zh:
            ic(em, text)
        text = text.replace(em, emoji2zh.get(em, '[UNK]'))
    if return_emojis:
        return text, emojis
    else:
        return text


def prepare_main(args):
    df_train = load_train_data()
    df_test = load_test_data()

    if args.build_emoji_vocab:
        build_emoji_vocab(df_train, df_test)
    # write2json({"[BOS]": 0, "[EOS]": 1}, "data/bos_eos_vocab.json")

    emoji2text, emoji2text_unique = build_emoji_map(df_train)

    # df_train = filter_unique_map(df_train, emoji2text_unique)

    df_train.to_csv("data/df_total.csv", index=False, sep=CSV_SEP)
    ic(df_train.shape)
    ic(df_train.head())
    ic(df_train["source_text"].str.len().describe())
    ic(df_train["target_text"].str.len().describe())

    # demojize = partial(emojiswitch.demojize, delimiters=(BOS_TOKEN, EOS_TOKEN), lang="zh")
    emoji2zh = {k: BOS_TOKEN + v[1:-1] + EOS_TOKEN for k, v in EMOJI_ZH.items()}
    emoji2text_cldr_tts = load_json("data/emoji2text_cldr_tts.json")
    emoji2zh.update(emoji2text_cldr_tts)
    emoji2zh.update(emoji2text_unique)
    emoji2zh.update(CUSTOM_EMOJI_MAP)
    write2json(emoji2zh, EMOJI2ZH_PATH)

    # FIXME: failure on data augmentation
    # df_mask = []
    # for idx in trange(df_train.shape[0]):
    #     source = df_train['source_text'][idx]
    #     target = df_train['target_text'][idx]
    #     source_replaced, emojis = cldr_demojize(source, emoji2zh, return_emojis=True)
    #     if (source_replaced == target):
    #         source_masked = source
    #         for em in emojis:
    #             source_masked = source_masked.replace(em, "[MASK]")
    #         df_mask.append({"source_text": source_masked, "target_text": target})
    #         # df_mask = pd.concat([df_mask, pd.DataFrame({"source_text": [source_masked], "target_text": [target]})])
    #     df_train['source_text'][idx] = source_replaced
    # df_mask = pd.DataFrame(df_mask)
    # ic(df_mask.shape)
    # df_train = pd.concat([df_train, df_mask], axis=0)

    df_train["source_text"] = df_train["source_text"].apply(lambda x: cldr_demojize(x, emoji2zh))
    df_train["source_text"] = df_train["source_text"].apply(lambda x: x.replace(" ", ""))
    ic(df_train.shape)
    df_train.drop_duplicates(subset=["source_text", "target_text"], inplace=True)
    ic(df_train.shape)
    ic(df_train.head())
    ic(df_train[df_train['source_text'] == df_train['target_text']].shape)
    ic(df_train[df_train['source_text'] != df_train['target_text']].shape)
    ic(df_train["source_text"].str.len().describe(percentiles=[0.95]))
    ic(df_train["target_text"].str.len().describe(percentiles=[0.95]))

    df_train.to_csv(args.output_file, index=False, sep=CSV_SEP)

    # df_test["prediction"] = df_test["prediction"].apply(process_text)
    # # df_test["prediction"] = df_test["prediction"].apply(full2half)

    # df_test["prediction"] = df_test["prediction"].apply(
    #     lambda x: cldr_demojize(x, emoji2zh, return_emojis=False))
    # df_test["prediction"] = df_test["prediction"].apply(lambda x: x.replace(" ", ""))

    # ic(df_test.head())
    # ic(df_test["prediction"].str.len().describe(percentiles=[0.95]))
    # df_test.to_csv("data/df_test_demojized.csv", index=False, sep=CSV_SEP)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", "-o", type=str, default="data/df_total_demojized.csv")
    parser.add_argument("--build_emoji_vocab", action="store_true", default=False)
    args = parser.parse_args()
    ic(args)
    prepare_main(args)
