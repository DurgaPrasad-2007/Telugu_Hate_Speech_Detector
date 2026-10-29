"""Run: python tests/test_despam.py"""
import os, sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from despam import SPLITS, flag_all, marker_flag

assert marker_flag(["మీ సమస్యలు పరిష్కారం", "వశీకరణం చేయబడును", "నాక్ నాక్ నాకొద్దు"]) == [True, True, False]
spam = "భార్యాభర్తల సమస్యకైనా ఫోన్ చేయండి 9999999999"
near = "భార్యాభర్తల సమస్యకైనా ఫోన్ చేయండి 8888888888"  # no extra marker needed; here marker present anyway
frames = {s: pd.DataFrame({"label": [0, 1], "text": [spam if s == "train" else near, "నాక్ నాక్ నాక్ నాకొద్దు"]}) for s in SPLITS}
flags, _, _ = flag_all(frames)
assert all(flags[s] == [True, False] for s in SPLITS), flags
print("ok")
