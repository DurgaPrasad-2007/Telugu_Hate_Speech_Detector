"""Knowledge distillation: teacher -> small student (docs/RESEARCH_REVIEW.md 4.5 Phase 4).
Loss = a*CE + b*T^2*KL(teacher_T || student_T) [+ c*MSE(hidden)], T=2.
Needs torch/transformers/datasets at train time; distill_loss() and the self-check are pure python.
Usage: python src/distill.py --teacher <dir> --student <hf-id> --out results/distill [--hidden-mse 1.0]
       python src/distill.py --selfcheck
NOT run end-to-end in the authoring session (torch not installed): training is untested.
"""
import argparse
import math
import os
import sys


def _softmax(z, T=1.0):
    m = max(z)
    e = [math.exp((x - m) / T) for x in z]
    s = sum(e)
    return [x / s for x in e]


def distill_loss(s_logits, t_logits, label, T=2.0, alpha=0.5, beta=0.5, gamma=0.0,
                 s_hidden=None, t_hidden=None):
    """Single-example loss on plain lists. Returns alpha*CE + beta*T^2*KL + gamma*MSE(hidden)."""
    ce = -math.log(_softmax(s_logits)[label])
    ps, pt = _softmax(s_logits, T), _softmax(t_logits, T)
    kl = sum(t * math.log(t / s) for t, s in zip(pt, ps) if t > 0)
    mse = 0.0
    if gamma and s_hidden is not None:
        mse = sum((a - b) ** 2 for a, b in zip(s_hidden, t_hidden)) / len(s_hidden)
    return alpha * ce + beta * T * T * kl + gamma * mse


def selfcheck():
    t = [3.0, -1.0]
    # student == teacher: KL term is 0, loss reduces to alpha*CE
    assert abs(distill_loss(t, t, 0, alpha=1, beta=1) - (-math.log(_softmax(t)[0]))) < 1e-12
    # worse student -> larger loss; KL non-negative
    assert distill_loss([-3.0, 3.0], t, 0) > distill_loss(t, t, 0)
    assert distill_loss([0.0, 0.0], t, 0, alpha=0, beta=1) > 0
    # hidden MSE adds exactly gamma*mse
    d = distill_loss(t, t, 0, alpha=0, beta=0, gamma=2.0, s_hidden=[1, 1], t_hidden=[0, 2])
    assert abs(d - 2.0) < 1e-12
    print("selfcheck ok")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--selfcheck", action="store_true")
    p.add_argument("--teacher"); p.add_argument("--student"); p.add_argument("--out", default="results/distill")
    p.add_argument("--domain", default="telugu"); p.add_argument("--preprocess", default="normalized")
    p.add_argument("--T", type=float, default=2.0)
    p.add_argument("--alpha", type=float, default=0.5); p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--hidden-mse", dest="gamma", type=float, default=0.0, help="weight; 0 = off")
    p.add_argument("--epochs", type=int, default=5); p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-5); p.add_argument("--max_length", type=int, default=96)
    a = p.parse_args()
    if a.selfcheck:
        return selfcheck()
    if not (a.teacher and a.student):
        p.error("--teacher and --student required")

    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
    sys.path.insert(0, os.path.dirname(__file__))
    from data_utils import load_dataset_dict

    # ponytail: teacher must be a full (merged) HF seq-cls dir sharing the student's tokenizer;
    # merge a LoRA adapter first (peft merge_and_unload) and save it.
    tok = AutoTokenizer.from_pretrained(a.student)
    teacher = AutoModelForSequenceClassification.from_pretrained(a.teacher).eval()
    student = AutoModelForSequenceClassification.from_pretrained(a.student, num_labels=teacher.config.num_labels)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    teacher.to(dev); student.to(dev)
    proj = None
    if a.gamma:
        proj = torch.nn.Linear(student.config.hidden_size, teacher.config.hidden_size).to(dev)

    ds = load_dataset_dict(domain=a.domain, preprocess=a.preprocess)
    enc = ds.map(lambda e: tok(e["text"], truncation=True, max_length=a.max_length), batched=True, remove_columns=["text"])
    dl = torch.utils.data.DataLoader(enc["train"], batch_size=a.bs, shuffle=True,
                                     collate_fn=DataCollatorWithPadding(tok))
    params = list(student.parameters()) + (list(proj.parameters()) if proj else [])
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    for ep in range(a.epochs):
        student.train()
        for b in dl:
            b = {k: v.to(dev) for k, v in b.items()}
            y = b.pop("labels")
            with torch.no_grad():
                t = teacher(**b, output_hidden_states=bool(a.gamma))
            s = student(**b, output_hidden_states=bool(a.gamma))
            # same formula as distill_loss()
            loss = a.alpha * F.cross_entropy(s.logits, y) + a.beta * a.T ** 2 * F.kl_div(
                F.log_softmax(s.logits / a.T, -1), F.softmax(t.logits / a.T, -1), reduction="batchmean")
            if a.gamma:  # last-layer CLS state, projected to teacher width
                loss = loss + a.gamma * F.mse_loss(proj(s.hidden_states[-1][:, 0]), t.hidden_states[-1][:, 0])
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"epoch {ep} last loss {loss.item():.4f}")
    os.makedirs(a.out, exist_ok=True)
    student.save_pretrained(a.out); tok.save_pretrained(a.out)


if __name__ == "__main__":
    main()
