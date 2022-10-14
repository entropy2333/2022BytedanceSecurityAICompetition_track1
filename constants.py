from emojiswitch.unicode_codes import EMOJI_ZH

GENERATION_KWARGS = dict(max_length=100,
                         num_beams=10,
                         do_sample=False,
                         top_k=50,
                         top_p=1.0,
                         early_stopping=False,
                         repetition_penalty=2.5)

EMOJI2ZH_PATH = "data/emoji2zh.json"

DF_TRAIN_RAW_PATH = "raw_data/emoji7w.xlsx"
DF_TEST_RAW_PATH = "raw_data/emoji7w-test_data.csv"

CSV_SEP = "\t"

BOS_TOKEN = ""  # ":" "[BOS]"
EOS_TOKEN = ""  # ":" "[BOS]"
TASK_PREFIX = ""