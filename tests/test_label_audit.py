import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from label_audit import cl_scores


def test_cl_scores():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 1])
    p = np.array([.9, .95, .1, .1, .05, .9, .1, .85])  # idx2 labeled abusive but p=.1; idx5 labeled non-abusive but p=.9
    s, f, t = cl_scores(p, y)
    assert f[2] and f[5] and f.sum() == 2
    assert set(np.argsort(-s)[:2]) == {2, 5}
    assert s[2] > 0 and s[0] < 0


if __name__ == "__main__":
    test_cl_scores()
    print("ok")
