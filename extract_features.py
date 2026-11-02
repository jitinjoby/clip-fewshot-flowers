import json, numpy as np, torch
from torch.utils.data import DataLoader
from torchvision import datasets
from transformers import CLIPModel, CLIPProcessor
from flower_names import names as all_names

ROOT = "flowers20"
MODEL = "openai/clip-vit-base-patch32"
dev = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained(MODEL).to(dev).eval()
proc = CLIPProcessor.from_pretrained(MODEL)

def prep(img):
    return proc(images=img, return_tensors="pt")["pixel_values"][0]

@torch.no_grad()
def extract(split):
    ds = datasets.ImageFolder(f"{ROOT}/{split}", transform=prep)
    dl = DataLoader(ds, batch_size=64, num_workers=0)
    feats, labels = [], []
    for x, y in dl:
        f = model.get_image_features(pixel_values=x.to(dev))
        if not isinstance(f, torch.Tensor):          # newer transformers versions
            f = f.pooler_output
        feats.append(torch.nn.functional.normalize(f, dim=-1).cpu())
        labels.append(y)
    return torch.cat(feats).numpy(), torch.cat(labels).numpy(), ds.classes

Xtr, ytr, classes = extract("train")
Xva, yva, c2 = extract("val")
Xte, yte, c3 = extract("test")
assert classes == c2 == c3, "class folders differ between splits"

np.savez("features.npz", Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, Xte=Xte, yte=yte)

# class names in label order 0..19 (class_005 -> names[4])
cls_names = [all_names[int(c.split("_")[1]) - 1] for c in classes]
json.dump(cls_names, open("class_names.json", "w"), indent=1)
print(cls_names)
print("train/val/test:", len(ytr), len(yva), len(yte))