# 2022 字节跳动安全AI挑战赛——Emoji复杂文本识别

团队名称：Anya
线上得分：0.87730（rank1）

## 赛题描述

- 比赛地址:[2022 字节跳动安全AI挑战赛](https://security.bytedance.com/fe/2022/ai-challenge#/challenge)
- 输入：含emoji的文本
- 输出：正确翻译后的文本
- 评价指标：BLEU和WER的加权

## Requirements

```bash
pip install -r requirements.txt
```

## Quick Start

### Rule-based baseline (score:0.83)

根据训练集的数据，保留出现频率大于等于2的emoji->text映射关系，构建基本规则字典`data/map_v5.json`和其子集`data/map_mostCommon.json`、`data/map_multi.json`；对于测试集中oov的emoji，通过cldr公开的`unicode-org.xml`构建oov.json作为补充。直接使用上述字典对测试集进行替换，得到`submit/test_map_v5.csv`即为baseline。

```bash
python build_map_v5.py
python predict_use_map.py
```

### Preprocess

将数据集 `emoji7w.xlsx` 和 `emoji7w-test_data.csv` 放在 `raw_data` 目录

```bash
python scripts/parse_cldr_xml.py
python data_process.py
```

将在 `data` 目录下生成 `df_total_demojized.csv` 用于训练


(可选) 对多次训练的模型预测集成投票，得到伪标签文件，`vote/vote.py`中记录了用于投票的若干模型预测；再根据伪标签生成训练文件 `data/df_total_demojized_pseudo.csv`，其中伪标签文件和提交文件格式一致

```bash
# cd vote
# python vote.py
# cd ..
python merge_pseudo_label.py -f ${pseudo_csv}
```

### Training

预训练模型采用 [fnlp/bart-large-chinese](https://huggingface.co/fnlp/bart-base-chinese)，机器资源紧张可以使用base版本或修改batch_size。

默认是全量训练，可以修改`train_with_total`参数采用验证集。

```bash
python train_bart.py \
    --df_train ${df_train} \
    --df_valid ${df_valid} \
    --train_with_total True \
    --model_type "bart" \
    --pretrain_path "fnlp/bart-large-chinese" \
    --output_dir "outputs" \
    --max_epochs 10 \
    --batch_size 64 \
    --learning_rate 0.00002 \
    --source_max_token_len 50 \
    --target_max_token_len 50
```

输出的模型在`${output_dir}`目录下，结构大致如下

```bash
outputs/
└── simplet5-epoch-9-train-loss-9.9999-val-loss-9.9999
    ├── config.json
    ├── pytorch_model.bin
    ├── special_tokens_map.json
    ├── tokenizer_config.json
    ├── tokenizer.json
    └── vocab.txt
```

### Inference

加载模型进行推理，并指定输出文件，默认在`submit`目录下。

```bash
python load_bart.py \
    --checkpoint ${checkpoint_dir} \
    --submit_file ${submit_file} \
    --batch_size 32
```

### Post Process

模型推理的Bad Cases存在缺字漏字现象，用Rule-based Baseline对该部分作修正，得到最终结果。

```bash
python banjiao2quanjiao.py --file ./submit/model/0927_bart_beam_large_pseudo_beam10.csv

python merge_map_model.py \
    --test ./data/emoji7w-test_data.csv \
    --map ./submit/test_map_v5.csv \
    --model ./submit/model/0927_bart_beam_large_pseudo_beam10_h2f.csv \
    --output ./submit/merge/merge_0922v5_bart0927beamlargepseudo_beam10.csv
```


## Acknowledgements

感谢 [pytorch-lightning](https://github.com/Lightning-AI/lightning) 和 [simpleT5](https://github.com/Shivanandroy/simpleT5)。

## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
        <td align="center"><a href="http://github.com/entropy2333"><img src="https://avatars.githubusercontent.com/u/40735723?v=4?s=70" width="70px;" alt=" entropy2333"/><br /><sub><b> entropy2333</b></sub></a><br /></td>
    </tr>
  </tbody>
</table>
