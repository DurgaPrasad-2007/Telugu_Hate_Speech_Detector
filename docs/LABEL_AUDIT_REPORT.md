# Label audit report

Data: THS_DATA_DIR=data/clean (spam-free). Internal labels 1=abusive (CSV 0=abusive). Pooled train+val n=24749, test n=2770. 5-fold stratified OOF LinearSVC (char_wb 2-5 + word 1-2 TF-IDF, C=0.3 chosen on val), Platt on OOF margins.

Per-class thresholds (mean P(class) among rows labeled that class): non-abusive 0.847, abusive 0.832. Flagged suspects: 687 (given abusive 418, given non-abusive 269).

MuRIL (`clean_muril-base-cased_lp8_lr5e-5`) has predictions only for val/test (it trained on train), so it is a held-out cross-check on val rows only: 2729 val rows, MuRIL disagrees with the given label on 290; among 72 OOF-flagged val rows it disagrees on 64.

## Test macro-F1 after acting on top-k TRAIN suspects

Untouched baseline (this script, same features, C=0.3): 0.8952 (reference 0.896). Val/test untouched, C fixed. Delta CI = paired bootstrap (10000 resamples, seed 0, same resamples for all rows, macro_f1_counts from significance.py). Control = removing k random train rows.

| k | action | test macro-F1 | delta vs baseline | 95% CI |
|---|---|---|---|---|
| 50 | remove | 0.8970 | +0.0018 | [-0.0014, +0.0051] |
| 50 | relabel | 0.8970 | +0.0018 | [-0.0014, +0.0051] |
| 50 | random-remove (control) | 0.8959 | +0.0007 | [-0.0007, +0.0022] |
| 100 | remove | 0.8974 | +0.0022 | [-0.0018, +0.0062] |
| 100 | relabel | 0.8970 | +0.0018 | [-0.0022, +0.0058] |
| 100 | random-remove (control) | 0.8952 | +0.0000 | [-0.0022, +0.0022] |
| 200 | remove | 0.8978 | +0.0025 | [-0.0018, +0.0069] |
| 200 | relabel | 0.8989 | +0.0036 | [-0.0010, +0.0084] |
| 200 | random-remove (control) | 0.8945 | -0.0008 | [-0.0033, +0.0018] |
| 400 | remove | 0.8952 | +0.0000 | [-0.0051, +0.0051] |
| 400 | relabel | 0.8956 | +0.0004 | [-0.0047, +0.0058] |
| 400 | random-remove (control) | 0.8970 | +0.0018 | [-0.0011, +0.0051] |

## Example suspects (sanity check; not human verified)

- #1 given non-abusive, p_abusive_oof=0.999: సెక్స్ గా ఉంది
- #2 given non-abusive, p_abusive_oof=0.998: సెక్స్ సి వీడియోస్
- #3 given non-abusive, p_abusive_oof=0.998: ఇలాంటి ప్రోగ్రాములు వల్ల ఆడవాళ్ళు ఎంత చెత్త ముండ లో తెలుస్తుంది
- #4 given non-abusive, p_abusive_oof=0.998: ఒరేయ్ బోకు యాదవ నీ బుర్ర ఎన్ని సారులు తీప్పు తావు రా ఎదవ నీ అమ్మ వీడియో
- #5 given non-abusive, p_abusive_oof=0.998: చి పో నాకు సిగ్గు 😆
- #6 given non-abusive, p_abusive_oof=0.998: అరే ఎరీ పూక అది నీ గుద్దల పెట్టుకున్న కూడ ఎలుగుతది రా
- #7 given non-abusive, p_abusive_oof=0.998: ఓసి నీ సిగ్గు సిగ్గు
- #8 given non-abusive, p_abusive_oof=0.997: కుక్క తిండి తింటుంది కుక్క
- #276 given abusive, p_abusive_oof=0.073: పనికి వచ్చే న్యూస్ లు పెట్టండి రా 🙄మూవీ హిట్ అయితే ఎలా అయినా డబ్బులు వస్తాయి రైతు నష్టపోతే ఆత్మహత్య చేసుకుంటున్నారు 😒రైతు ఉంటే నే ఇండస్ట్రీ 
- #173 given non-abusive, p_abusive_oof=0.952: బొడ్డు చూపించు ప్లీస్
- #143 given abusive, p_abusive_oof=0.043: నీ సామాన్లు చాలా 👌👌👌👌
- #208 given abusive, p_abusive_oof=0.056: నా ముద్దుల రౌడీ పిల్లకి నా ముద్దుల బంగారం కి నా ముద్దల మార్దల్ కి గుడ్ మార్నింగ్ ర మై లవ్లీ సిరి బంగారం 😘😘😘😘😘😘😘😘😘😘😘😘❤❤❤😘😘😘😘
- #205 given abusive, p_abusive_oof=0.055: నాతో వార్ నీకే డేంజర్ 😎
- #212 given abusive, p_abusive_oof=0.057: దండం రా సామీ నీకు...
- #249 given abusive, p_abusive_oof=0.065: గుండు గికిచుకొ చెప్పిన

## Caveats

- Removing/relabeling on auto-suspects is an unreviewed heuristic: no human has confirmed any suspect is a true mislabel. Relabeling assumes the model is right. The deltas say nothing about label precision.
- Suspects come from a TF-IDF model, so they are biased toward what bag-of-ngrams cannot fit.
- `label_audit_gold_review.csv` (top 300) has a blank `human_label` column (use CSV convention 0=abusive); label precision of the top 300 is unmeasured until it is filled.
