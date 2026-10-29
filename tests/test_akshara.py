import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from akshara import normalize, segment
from translit import phonetic_key, to_telugu

assert segment("నన్ను") == ["న", "న్ను"] and segment("చచ్చి") == ["చ", "చ్చి"]
assert normalize("వె‍ధ‍వ") == "వెధవ"
assert normalize("వె.ధ.వ") == "వెధవ" and normalize("వె-ధ-వ") == "వెధవ"
assert normalize("d*e*n*g*u") == "dengu" and normalize("v e d h a") == "vedha"
assert normalize("వె ధ వ ల ని") == "వెధవలని"
for s in ["నన్ను చచ్చి", "వె**వ", "a b c", "a.b", "good-day", "నా ని ఈ", "బాగుంది.నువ్వు", "వాళ్లలో ఇంకా"]:
    assert normalize(s) == s, s
assert phonetic_key("vaddu") == phonetic_key("voddu") == phonetic_key("woddu")
assert to_telugu("vaddu") == to_telugu("voddu") == to_telugu("woddu") == "వద్దు"
print("akshara tests OK")

# spaced short slurs (lexicon from train, src/abusive_lexicon.txt): rejoin only known abusive tokens
assert normalize("నువ్వు వె ధ వ") == "నువ్వు వెధవ" and normalize("వీడు పెద్ద స న్నా సి") == "వీడు పెద్ద సన్నాసి"
assert normalize("ఓరి నా కొ డ కా") == "ఓరి నాకొడకా"
for s in ["నా ని ఈ", "అ ది ని", "ఆ రో జు", "మీ రు ఎ ప్పు"]:  # unknown 3-4 unit runs stay
    assert normalize(s) == s, s
print("akshara short-slur tests OK")
