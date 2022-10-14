#!/bin/bash

set -x
set -e
python banjiao2quanjiao.py --file ./submit/model/0927_bart_beam_large_pseudo_beam10.csv
python banjiao2quanjiao.py --file ./submit/model/0927_bart_beam_large_pseudo_mean3.csv
python banjiao2quanjiao.py --file ./submit/model/0927_bart_beam_large_pseudo_mean5.csv


python merge_map_model.py \
    --test ./data/emoji7w-test_data.csv \
    --map ./submit/test_map_v5.csv \
    --model ./submit/model/0927_bart_beam_large_pseudo_beam10_h2f.csv \
    --output ./submit/merge/merge_0922v5_bart0927beamlargepseudo_beam10.csv

python merge_map_model.py \
    --test ./data/emoji7w-test_data.csv \
    --map ./submit/test_map_v5.csv \
    --model ./submit/model/0927_bart_beam_large_pseudo_mean3_h2f.csv \
    --output ./submit/merge/merge_0922v5_bart0927beamlargepseudo_mean3.csv

python merge_map_model.py \
    --test ./data/emoji7w-test_data.csv \
    --map ./submit/test_map_v5.csv \
    --model ./submit/model/0927_bart_beam_large_pseudo_mean5_h2f.csv \
    --output ./submit/merge/merge_0922v5_bart0927beamlargepseudo_mean5.csv
