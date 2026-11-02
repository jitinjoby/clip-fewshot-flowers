import random, shutil, json
from pathlib import Path
from scipy.io import loadmat

SEED = 42
SRC = Path("jpg")
DST = Path("flowers20")

labels = loadmat("imagelabels.mat")["labels"].flatten()   # length 8189, values 1..102
setid  = loadmat("setid.mat")
splits = {
    "train": setid["trnid"].flatten(),   # 1020 image ids
    "val":   setid["valid"].flatten(),   # 1020 image ids
    "test":  setid["tstid"].flatten(),   # 6149 image ids
}

# 1) pick 20 of the 102 classes (reproducible)
random.seed(SEED)
chosen = sorted(random.sample(range(1, 103), 20))
print("Chosen classes:", chosen)
json.dump(chosen, open("chosen_classes.json", "w"))

# 2) copy images into flowers20/{split}/{class}/
counts = {}
for split, ids in splits.items():
    for img_id in ids:                       # img_id is 1-indexed
        cls = int(labels[img_id - 1])        # labels array is 0-indexed
        if cls not in chosen:
            continue
        out_dir = DST / split / f"class_{cls:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(SRC / f"image_{img_id:05d}.jpg", out_dir)
        counts[split] = counts.get(split, 0) + 1

print(counts)