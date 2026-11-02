import json, numpy as np, torch
from transformers import CLIPModel, CLIPProcessor
from sklearn.metrics import accuracy_score, f1_score

MODEL = "openai/clip-vit-base-patch32"
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained(MODEL).to(dev).eval()
proc = CLIPProcessor.from_pretrained(MODEL)

d = np.load("features.npz")
Xva, yva, Xte, yte = d["Xva"], d["yva"], d["Xte"], d["yte"]
cls_names = json.load(open("class_names.json"))

@torch.no_grad()
def text_feats(template):
    prompts = [template.format(n) for n in cls_names]
    t = proc(text=prompts, return_tensors="pt", padding=True).to(dev)
    f = model.get_text_features(**t)
    if not isinstance(f, torch.Tensor):              # newer transformers versions
        f = f.pooler_output
    return torch.nn.functional.normalize(f, dim=-1).cpu().numpy()

templates = ["{}", "a photo of a {}.", "a photo of a {}, a type of flower."]

# choose the prompt on VAL
best_t, best_acc = None, -1
for t in templates:
    acc = accuracy_score(yva, (Xva @ text_feats(t).T).argmax(1))
    print(f"val acc {acc:.3f} | {t}")
    if acc > best_acc:
        best_t, best_acc = t, acc

# report on TEST
pred = (Xte @ text_feats(best_t).T).argmax(1)
res = {"template": best_t,
       "test_acc": accuracy_score(yte, pred),
       "test_macro_f1": f1_score(yte, pred, average="macro")}
print(res)
json.dump(res, open("zero_shot_result.json", "w"), indent=1)