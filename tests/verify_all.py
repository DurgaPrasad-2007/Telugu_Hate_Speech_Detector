import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding='utf-8')
import time
from src.infer import HateSpeechClassifier

test_cases = [
    ("Clean Telugu 1", "ఈ సినిమా కథ, దర్శకత్వం చాలా అద్భుతంగా ఉన్నాయి."),
    ("Clean Telugu 2", "నమస్కారం అండి, మీరు ఎలా ఉన్నారు?"),
    ("Abusive Telugu 1", "నువ్వు ఒక పనికిమాలిన చెత్త నాకొడుకువి."),
    ("Abusive Telugu 2", "లంజాకొడకా నీకు బుద్ధి లేదా రా."),
    ("Spaced Obfuscated", "వెంకటేష్ ఒక వె ధ వ"),
    ("Astrology Spam Check", "మీ సమస్యలకు పరిహారం వశీకరణం చేయబడును ఫోన్ చేయండి")
]

for model_name, path in [('INT8 ONNX Engine (Real-Time)', 'results/final_best/onnx_int8'), ('PyTorch LoRA Adapter', 'models/best')]:
    print(f"==============================")
    print(f"TESTING: {model_name}")
    print(f"Path: {path}")
    print(f"==============================")
    t0 = time.time()
    clf = HateSpeechClassifier(model_dir=path)
    init_time = (time.time() - t0) * 1000
    print(f"Initialization time: {init_time:.2f} ms\n")
    
    for desc, text in test_cases:
        t_start = time.perf_counter()
        res = clf.predict([text])[0]
        latency = (time.perf_counter() - t_start) * 1000
        print(f"[{desc}]")
        print(f"  Input: \"{text}\"")
        print(f"  Decision: {res['decision'].upper()} | Label: {res['label']} | P(Abuse): {res['probability']:.4f} | Latency: {latency:.2f} ms")
        if res['probability'] > 0.5:
            expl = clf.explain(text)
            flagged = [w for w, score, flag in expl if score > 0.005 or flag]
            print(f"  Attribution Drivers: {flagged}")
        print()
