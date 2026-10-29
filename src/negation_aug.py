"""Abusive-token lexicon (TRAIN only) + Telugu negation / reported-speech training augmentation.

* mine_lexicon(): tokens frequent in abusive train rows and (almost) absent from non-abusive ones.
  `--write-lexicon` stores them in src/abusive_lexicon.txt (used by akshara.normalize for spaced slurs).
* `--build` writes data/clean_aug/: train = clean train + augmentation, val/test = untouched copies.
  Negated/denied/prohibited slur -> NON-abusive; matching affirmative frames with the same slur -> abusive,
  so the model learns the contrast and not 'negation => safe'.
Labels: CSV 0=abusive, 1=non-abusive (raw); internal 1=abusive (data_utils flips on load). Rows are written RAW.
All templates are MODEL-AUTHORED, not native-speaker reviewed. They are deliberately different sentence frames
from tests/functional/telugu_hatecheck.csv (checked in tests/test_negation_aug.py: no shared text and no shared
word 4-gram once slurs are masked) so that file stays a fair holdout.
usage: THS_DATA_DIR=data/clean python src/negation_aug.py --write-lexicon --build"""
import collections, os, random, re, shutil, sys
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SRC, OUT = os.path.join(ROOT, "data", "clean"), os.path.join(ROOT, "data", "clean_aug")
LEX = os.path.join(HERE, "abusive_lexicon.txt")
HATECHECK = os.path.join(ROOT, "tests", "functional", "telugu_hatecheck.csv")
TELUGU = re.compile(r"[ఀ-౿]+")
MIN_ABUSIVE, MAX_NON_RATIO, N_SLURS, SEED = 15, 0.05, 60, 0

# mined tokens that are names, topics, identity terms or ordinary words rather than epithets: not used as template slurs
DENY = set("ఘోస్ట్ రావణ్ తనీష్ కుషాల్ యమ్మన్ పెయింటర్ లేబర్ టీడీపీ కారం బజారు బజార్ లారా రాంకు మాదిగ యాదవ తురుక గొల్లి కుంటి గుడ్డి "
           "పిచ్చి పోసి కోసి నోట్లో దాన తూ పగల నల్ల నల్లటి మాడ లేత పండ శరం నెత్తురు చిల్లర కుక్కలు మూసుకో గుడ్డలు పేస్టు పేస్టులు "
           "తొడలు తొడల నరాలు రొమ్ములు లేపి తోసి చెప్పుతో సల్లు బే పీతి గజ్జి దూర్చి దూల దుల".split())
# (negated/denied/prohibited -> non-abusive, affirmative control -> abusive); {S}=slur, {P}=person (accusative)
PAIRS = [
    ("నేను {P} {S} అని పిలవలేదు", "నేను {P} {S} అని పిలిచాను"),
    ("{P} {S} అని పిలవడం తప్పు", "{P} {S} అని పిలవడం సరైనదే"),
    ("{S} అనే మాట నేను ఎప్పుడూ వాడను", "{S} అనే మాట నీకు బాగా సరిపోతుంది"),
    ("{P} {S} అని తిట్టడం నాకు నచ్చదు", "{P} {S} అని తిట్టడం నాకు చాలా ఇష్టం"),
    ("అసలు ఆ మనిషి {S} కాదని అందరికీ తెలుసు", "అసలు ఆ మనిషి {S} అని అందరికీ తెలుసు"),
    ("{S} అని తిట్టే అలవాటు నాకు లేదు", "{S} అని తిట్టే అలవాటు నాకు ఉంది"),
    ("దయచేసి ఎవరినీ {S} అని పిలవకండి", "కావాలంటే మీరు కూడా వాడిని {S} అని పిలవండి"),
    ("నేను {S} అన్నానని అబద్ధం చెప్పారు", "నేను {S} అన్నాను, అది నిజమే"),
]
PERSON = ["అతన్ని", "ఆమెను", "వాళ్ళను", "మిమ్మల్ని", "నిన్ను"]
# slur-free hand-written denial / threat-denial (non-abusive) and threat controls (abusive)
NEG_FREE = ["నేను ఎప్పుడూ ఎవరినీ దూషించను", "ఇక్కడ ఎవరినీ కించపరచాలని నా ఉద్దేశం లేదు", "అసభ్యంగా మాట్లాడే అలవాటు నాకు లేదు",
            "నేను ఎవరికీ హాని చేయాలనుకోలేదు", "మీకు ఏ హాని చేయను, భయపడకండి", "ఎవరినైనా బెదిరించడం నా పద్ధతి కాదు",
            "ఆ మాట నేను అనలేదు, ఎవరో కల్పించారు", "అలా దూషించడం నాకు అసలు ఇష్టం ఉండదు"]
POS_FREE = ["మీ అందరికీ హాని చేసి తీరతాను", "ఎవరినీ వదలను, అందరినీ బెదిరిస్తాను", "అసభ్యంగా తిట్టడమే నా పద్ధతి",
            "నువ్వు కనిపిస్తే నీ అంతు చూస్తాను", "ఈ రోజు నిన్ను కొట్టి తీరతాను", "వాడి తల పగలగొడతాను",
            "మీ ఇంటికొచ్చి మిమ్మల్ని చంపుతాను", "నిన్ను ఇక్కడే నరికేస్తాను, చూసుకో"]


def mine_lexicon(train_raw):
    """train_raw: DataFrame with internal labels (1=abusive). Pure-Telugu tokens, doc-frequency based."""
    a, n = collections.Counter(), collections.Counter()
    for t, y in zip(train_raw.text, train_raw.label):
        for w in set(t.split()):
            if TELUGU.fullmatch(w):
                (a if y == 1 else n)[w] += 1
    return sorted([(w, c, n[w]) for w, c in a.items() if c >= MIN_ABUSIVE and n[w] <= MAX_NON_RATIO * c], key=lambda r: -r[1])


def pick_slurs(lex):
    return [w for w, _, _ in lex if w not in DENY][:N_SLURS]


def generate(slurs, seed=SEED):
    """-> list of (text, internal_label). Equal numbers of negated (0) and affirmative (1) slur sentences."""
    rng, out = random.Random(seed), []
    for s in slurs:
        for neg, pos in PAIRS:
            p = rng.choice(PERSON)
            out += [(neg.format(S=s, P=p), 0), (pos.format(S=s, P=p), 1)]
    out += [(t, 0) for t in NEG_FREE] + [(t, 1) for t in POS_FREE]
    seen, uniq = set(), []
    for t, y in out:
        if t not in seen:
            seen.add(t); uniq.append((t, y))
    return uniq


def fit_tfidf(data_dir, out_path):
    """Deployed TF-IDF recipe (src/baselines_classical.py: char_wb 2-5 + word 1-2, LinearSVC C=0.3) on data_dir train."""
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer as T
    from sklearn.pipeline import FeatureUnion, Pipeline
    from sklearn.svm import LinearSVC
    import data_utils
    data_utils.DATA_DIR = data_dir
    tr = data_utils.load_split("telugu", "train", "normalized")
    pipe = Pipeline([("features", FeatureUnion([
        ("char", T(analyzer="char_wb", ngram_range=(2, 5), max_features=300000, sublinear_tf=True)),
        ("word", T(analyzer="word", token_pattern=r"\S+", ngram_range=(1, 2), max_features=300000, sublinear_tf=True))])),
        ("clf", LinearSVC(C=0.3, max_iter=10000))]).fit(tr.text, tr.label)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(pipe, out_path)
    print("fit", len(tr), "->", out_path)


def main():
    from data_utils import load_split
    flags = sys.argv[1:]
    tr = load_split("telugu", "train", "normalized")
    lex = mine_lexicon(tr)
    if "--write-lexicon" in flags:
        open(LEX, "w", encoding="utf-8").write("\n".join(w for w, _, _ in lex) + "\n")
        print("lexicon tokens:", len(lex))
    if "--fit-tfidf" in flags:  # candidates go to results/aug_tfidf/, deployed models/tfidf is never touched
        for name, d in (("aug", OUT), ("clean_retrain", SRC)):
            fit_tfidf(d, os.path.join(ROOT, "results", "aug_tfidf", name + ".joblib"))
    if "--build" in flags:
        slurs = pick_slurs(lex)
        rows = generate(slurs)
        hold = set(pd.read_csv(HATECHECK, encoding="utf-8").text)
        others = set()
        for f in ("telugu_val.csv", "telugu_test.csv"):
            others |= set(pd.read_csv(os.path.join(SRC, f), encoding="utf-8").text.astype(str).str.strip())
        assert not any(t in hold or t in others for t, _ in rows), "augmentation overlaps holdout/val/test"
        os.makedirs(OUT, exist_ok=True)
        for f in ("telugu_val.csv", "telugu_test.csv"):
            shutil.copyfile(os.path.join(SRC, f), os.path.join(OUT, f))
        base = pd.read_csv(os.path.join(SRC, "telugu_train.csv"), encoding="utf-8")
        aug = pd.DataFrame({"label": [0 if y == 1 else 1 for _, y in rows], "text": [t for t, _ in rows]})  # RAW: 0=abusive
        pd.concat([base[["label", "text"]], aug], ignore_index=True).to_csv(os.path.join(OUT, "telugu_train.csv"), index=False, encoding="utf-8")
        print(f"slurs={len(slurs)} aug rows={len(rows)} (abusive {sum(y for _, y in rows)}, non {sum(1 - y for _, y in rows)}) "
              f"train {len(base)} -> {len(base) + len(aug)}")


if __name__ == "__main__":
    main()
