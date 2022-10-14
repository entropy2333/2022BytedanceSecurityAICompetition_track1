import pandas as pd
from collections import Counter, defaultdict
import copy
import os
import re
raw_test = pd.read_csv("../data/emoji7w-test_data.csv", sep="\t")

res_dict = dict()

# res_dict['res84409'] = pd.read_csv("../submit/merge_0917v4_t5fulltrainh2f.csv", sep="\t")
# res_dict['res84190'] = pd.read_csv("../submit/test_t5_full_train.csv", sep="\t")
# res_dict['res84091'] = pd.read_csv("../submit/test_t5_full_en.csv", sep="\t")
# res_dict['res83890'] = pd.read_csv("../submit/test_t5_prefix.csv", sep="\t")
# res_dict['res83692'] = pd.read_csv("../submit/new_res2.csv", sep="\t")
# res_dict['res85004'] = pd.read_csv("../submit/vote_res84409_res84190_res84091_res83890_res83692_1.csv", sep="\t")
# res_dict['res85169'] = pd.read_csv("../submit/merge_0917v4_t5from0916v3resumeh2f.csv", sep="\t")
# res_dict['res85219'] = pd.read_csv("../submit/merge_0917v4_t5pseudofullge4h2f.csv", sep="\t")
# res_dict['res85494'] = pd.read_csv("../submit/merge_0917v4_t5pseudofrom0916v3resumeh2f.csv", sep="\t")
# res_dict['res85304'] = pd.read_csv("../submit/merge_0917v4_t5hasmultimaph2f.csv", sep="\t")
# res_dict['res84529'] = pd.read_csv("../submit/merge_0917v4_t5augfrom0916v3h2f.csv", sep="\t")
# res_dict['res85620'] = pd.read_csv("../submit/merge_0917v4_t5hasmultimaph2f_1-1&oov.csv", sep="\t")
# res_dict['res85768'] = pd.read_csv("../submit/merge_0917v4_t5pseudofrom0916v3resumeh2f_1-1&oov.csv", sep="\t")
# res_dict['res85960'] = pd.read_csv("../submit/merge/merge_0917v4_t5pseudo2.csv", sep="\t")
# res_dict['res85830'] = pd.read_csv("../submit/t5_pseudov2.1_from_0916_v3_resume__h2f.csv", sep="\t")
# res_dict['res85974'] = pd.read_csv("../submit/merge/merge_0917v4_t5pseudov2.1h2f.csv", sep="\t")
# res_dict['res85867'] = pd.read_csv("../submit/merge/merge_0917v4_t5pseudov2.2h2f.csv", sep="\t")
# res_dict['res86210'] = pd.read_csv("../submit/merge/merge_0917v4_bart0920beam.csv", sep="\t")
# res_dict['res86303'] = pd.read_csv("../submit/merge/merge_0917v4_bartpseudov1.csv", sep="\t")
# res_dict['res86500'] = pd.read_csv("../submit/merge/merge_0917v4_bart0921beamlarge.csv", sep="\t")
# res_dict['res86692'] = pd.read_csv("../submit/merge/merge_0917v4_bart0921beamlargepseudo.csv", sep="\t")
# res_dict['res86516'] = pd.read_csv("../submit/merge/merge_0917v4_bart0921beamlargepseudov2.csv", sep="\t")
#### 
# res_dict['res87078'] = pd.read_csv("../submit/merge/merge_0922v5_bart0925beamlargepseudofgm4k.csv", sep="\t")
# res_dict['res87183'] = pd.read_csv("../submit/merge/merge_0922v5_bart0925beamlargepseudo.csv", sep="\t")
# res_dict['res87528'] = pd.read_csv("../submit/merge/merge_0922v5_bart0926beamlargepseudo.csv", sep="\t")
# res_dict['res87699'] = pd.read_csv("../submit/merge/merge_0922v5_bart0927beamlargepseudo.csv", sep="\t")
# res_dict['res87730'] = pd.read_csv("../submit/merge/merge_0922v5_bart0927beamlargepseudo_beam10.csv", sep="\t")
# res_dict['res87718'] = pd.read_csv("../submit/merge/merge_0922v5_bart0927beamlargepseudo_mean3.csv", sep="\t")
# res_dict['res87712'] = pd.read_csv("../submit/merge/merge_0922v5_bart0927beamlargepseudo_mean5.csv", sep="\t")

#### final
# res_dict['res87078'] = pd.read_csv("../submit/0925_bart_beam_large_pseudo_fgm4k_h2f.csv", sep="\t")
# res_dict['res87183'] = pd.read_csv("../submit/0925_bart_beam_large_pseudo_h2f.csv", sep="\t")
# res_dict['res87528'] = pd.read_csv("../submit/0926_bart_beam_large_pseudo_h2f.csv", sep="\t")
res_dict['res87699'] = pd.read_csv("../submit/0927_bart_beam_large_pseudo_h2f.csv", sep="\t")
res_dict['res87730'] = pd.read_csv("../submit/0927_bart_beam_large_pseudo_beam10_h2f.csv", sep="\t")
res_dict['res87718'] = pd.read_csv("../submit/0927_bart_beam_large_pseudo_mean3_h2f.csv", sep="\t")
res_dict['res87712'] = pd.read_csv("../submit/0927_bart_beam_large_pseudo_mean5_h2f.csv", sep="\t")


res_scores = sorted(list(res_dict.keys()), key=lambda x:-int(x[3:]))
highest = res_scores[-1]
print(res_scores)
if not os.path.exists(f"../data/{'_'.join(res_scores)}/"):
    os.mkdir(f"../data/{'_'.join(res_scores)}/")
if not os.path.exists(f"./{'_'.join(res_scores)}/"):
    os.mkdir(f"./{'_'.join(res_scores)}/")

vote_res = copy.deepcopy(raw_test)
ge3 = copy.deepcopy(raw_test)
ge4 = copy.deepcopy(raw_test)
ge5 = copy.deepcopy(raw_test)
ge6 = copy.deepcopy(raw_test)

vote_cnt = defaultdict(int)

for idx  in range(len(raw_test['prediction'])):
    preds = [df['prediction'][idx] for r_s, df in res_dict.items()]
    most_common = Counter(preds).most_common(2)
    vote_pred, cnt = most_common[0]
    if cnt >= 3:
        ge3['prediction'][idx] = vote_pred
    else:
        ge3['prediction'][idx] = "####"
    if cnt >= 4:
        ge4['prediction'][idx] = vote_pred
    else:
        ge4['prediction'][idx] = "####"
    if cnt >= 5:
        ge5['prediction'][idx] = vote_pred
    else:
        ge5['prediction'][idx] = "####"
    if cnt >= 6:
        ge6['prediction'][idx] = vote_pred
    else:
        ge6['prediction'][idx] = "####"
    
    if len(most_common) == 2 and most_common[0][1] == most_common[1][1] == 2:
        if res_dict[highest]['prediction'][idx] in most_common[1][0]:
            vote_pred, cnt = most_common[1]
    # if len(most_common) == 3: #and most_common[0][1] == most_common[1][1] == 1 :#== most_common[2][1] == 2:
    #     vote_pred, cnt = res_dict[highest]['prediction'][idx], 1
        # if res_dict[highest]['prediction'][idx] in most_common[1][0]:
        #     vote_pred, cnt = most_common[1]
        # elif res_dict[highest]['prediction'][idx] in most_common[2][0]:
        #     vote_pred, cnt = most_common[2]
    
    vote_cnt[cnt] += 1

    vote_pred = res_dict[highest]['prediction'][idx] if cnt == 1 else vote_pred
    ## 
    raw = raw_test['prediction'][idx]
    raw_ch = re.sub("([^\u4E00-\u9FA5]+)","",raw)
    vote_pred_ch = re.sub("([^\u4E00-\u9FA5]+)","",vote_pred)
    if len(vote_pred_ch) < len(raw_ch) or vote_pred_ch == raw_ch:
        vote_pred = res_dict[highest]['prediction'][idx]
        # for this_score in res_scores:
        #     this_pred = res_dict[this_score]['prediction'][idx]
        #     this_pred_ch = re.sub("([^\u4E00-\u9FA5]+)","", this_pred)
        #     if len(this_pred_ch) > len(raw_ch):
        #         print(raw, vote_pred, this_pred)
        #         vote_pred = this_pred
        #         break
        # for this_score in res_scores:
        #     this_pred = res_dict[this_score]['prediction'][idx]
        #     this_pred_ch = re.sub("([^\u4E00-\u9FA5]+)","", this_pred)
        #     if len(this_pred_ch) >= len(raw_ch) and this_pred_ch != raw_ch:
        #         print(raw, vote_pred, this_pred)
        #         vote_pred = this_pred
        #         break
        # print(raw_test['id'][idx], raw_ch, vote_pred_ch,)

    vote_res['prediction'][idx] = vote_pred


print(f"vote_cnt: {vote_cnt}")
vote_res.to_csv(f"./{'_'.join(res_scores)}/vote.csv", sep="\t", index=False)

ge3[ge3['prediction'] !="####"].to_csv(f"./{'_'.join(res_scores)}/pesudo_ge3.csv", sep="\t", index=False)
ge4[ge4['prediction'] !="####"].to_csv(f"./{'_'.join(res_scores)}/pesudo_ge4.csv", sep="\t", index=False)
ge5[ge5['prediction'] !="####"].to_csv(f"./{'_'.join(res_scores)}/pesudo_ge5.csv", sep="\t", index=False)
ge6[ge6['prediction'] !="####"].to_csv(f"./{'_'.join(res_scores)}/pesudo_ge6.csv", sep="\t", index=False)

res_map = pd.read_csv("../submit/test_map_v5.csv", sep="\t")



ge3_data_t5 = list()
ge3_data_t5_fromraw = list()
for idx, id in enumerate(ge3['id']):
    pred_map = res_map['prediction'][idx]
    pred_map_fromraw = res_map_fromraw['prediction'][idx]
    pseudo = ge3['prediction'][idx]
    if pred_map != pseudo or 1:
        ge3_data_t5.append([pred_map, pseudo])
        ge3_data_t5_fromraw.append([pred_map_fromraw, pseudo])
ge3_t5 = pd.DataFrame(columns=['source_text', 'target_text'],data=ge3_data_t5)
ge3_t5[ge3_t5['target_text'] !="####"].to_csv(f"../data/{'_'.join(res_scores)}/pseudo_ge3.csv", sep="\t", index=False)
print(ge3_t5[ge3_t5['target_text'] !="####"].shape)

ge4_data_t5 = list()
ge4_data_t5_fromraw = list()
for idx, id in enumerate(ge4['id']):
    pred_map = res_map['prediction'][idx]
    pred_map_fromraw = res_map_fromraw['prediction'][idx]
    pseudo = ge4['prediction'][idx]
    if pred_map != pseudo or 1:
        ge4_data_t5.append([pred_map, pseudo])
        ge4_data_t5_fromraw.append([pred_map_fromraw, pseudo])
ge4_t5 = pd.DataFrame(columns=['source_text', 'target_text'],data=ge4_data_t5)
ge4_t5[ge4_t5['target_text'] !="####"].to_csv(f"../data/{'_'.join(res_scores)}/pseudo_ge4.csv", sep="\t", index=False)
print(ge4_t5[ge4_t5['target_text'] !="####"].shape)

ge5_data_t5 = list()
ge5_data_t5_fromraw = list()
for idx, id in enumerate(ge5['id']):
    pred_map = res_map['prediction'][idx]
    pred_map_fromraw = res_map_fromraw['prediction'][idx]
    pseudo = ge5['prediction'][idx]
    if pred_map != pseudo or 1:
        ge5_data_t5.append([pred_map, pseudo])
        ge5_data_t5_fromraw.append([pred_map_fromraw, pseudo])
ge5_t5 = pd.DataFrame(columns=['source_text', 'target_text'],data=ge5_data_t5)
ge5_t5[ge5_t5['target_text'] !="####"].to_csv(f"../data/{'_'.join(res_scores)}/pseudo_ge5.csv", sep="\t", index=False)
print(ge5_t5[ge5_t5['target_text'] !="####"].shape)

ge6_data_t5 = list()
ge6_data_t5_fromraw = list()
for idx, id in enumerate(ge6['id']):
    pred_map = res_map['prediction'][idx]
    pred_map_fromraw = res_map_fromraw['prediction'][idx]
    pseudo = ge6['prediction'][idx]
    if pred_map != pseudo or 1:
        ge6_data_t5.append([pred_map, pseudo])
        ge6_data_t5_fromraw.append([pred_map_fromraw, pseudo])
ge6_t5 = pd.DataFrame(columns=['source_text', 'target_text'],data=ge6_data_t5)
ge6_t5[ge6_t5['target_text'] !="####"].to_csv(f"../data/{'_'.join(res_scores)}/pseudo_ge6.csv", sep="\t", index=False)
print(ge6_t5[ge6_t5['target_text'] !="####"].shape)