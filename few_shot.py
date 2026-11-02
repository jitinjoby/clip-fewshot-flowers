import json, numpy as np, csv
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

d = np.load("features.npz")
Xtr, ytr, Xte, yte = d["Xtr"], d["ytr"], d["Xte"], d["yte"]
zs = json.load(open("zero_shot_result.json"))

def sample_k(y, k, seed):
    rng = np.random.RandomState(seed)
    return np.concatenate([rng.choice(np.where(y == c)[0], k, replace=False)
                           for c in range(20)])

KS, SEEDS = [1, 2, 4, 8, 10], range(5)
rows = []
for k in KS:
    accs, f1s = [], []
    for s in SEEDS:
        idx = sample_k(ytr, k, s)
        clf = LogisticRegression(C=10, max_iter=2000)   # fixed C for all k
        clf.fit(Xtr[idx], ytr[idx])
        p = clf.predict(Xte)
        accs.append(accuracy_score(yte, p))
        f1s.append(f1_score(yte, p, average="macro"))
    rows.append([k, np.mean(accs), np.std(accs), np.mean(f1s), np.std(f1s)])
    print(rows[-1])

with open("results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["k", "acc_mean", "acc_std", "f1_mean", "f1_std"])
    w.writerows(rows)

r = np.array(rows)
plt.errorbar(r[:, 0], r[:, 1], r[:, 2], marker="o", label="Few-shot accuracy")
plt.errorbar(r[:, 0], r[:, 3], r[:, 4], marker="s", label="Few-shot macro-F1")
plt.axhline(zs["test_acc"], color="gray", ls="--", label="Zero-shot accuracy")
plt.axhline(zs["test_macro_f1"], color="gray", ls=":", label="Zero-shot macro-F1")
plt.xlabel("Labeled images per class (k)"); plt.ylabel("Test score")
plt.legend(); plt.grid(alpha=0.3)
plt.savefig("accuracy_vs_k.png", dpi=200, bbox_inches="tight")