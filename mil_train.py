# mil_train.py
import json, random
from glob import glob
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from mil_utils import load_embeddings_npz, parse_anuraset_strong, events_to_weak_and_time, build_label_index
from your_poolers import PoolingHead  # your class

class EmbeddingBagDataset(Dataset):
    def __init__(self, npz_paths, strong_map, label_index):
        self.npz_paths = npz_paths
        self.strong_map = strong_map      # dict: stem -> path to strong txt
        self.label_index = label_index
        self.C = len(label_index)
    def __len__(self): return len(self.npz_paths)
    def __getitem__(self, i):
        npz = self.npz_paths[i]
        E, s, e = load_embeddings_npz(npz)
        # map NPZ -> matching strong-labels path; adapt this to your naming
        stem = Path(npz).stem.replace(".embeddings","")
        txt = self.strong_map.get(stem, None)
        events = parse_anuraset_strong(txt) if txt and Path(txt).exists() else []
        weak, time_targets = events_to_weak_and_time(events, self.label_index, s, e)
        return torch.from_numpy(E), torch.from_numpy(weak), torch.from_numpy(time_targets), (s,e)

def collate(batch):
    # Ragged T? Here we assume all bags already 1 s chunks; pad if needed
    E_list, w_list, tmask_list, times = zip(*batch)
    # pad to max T
    T = max(E.shape[0] for E in E_list)
    D = E_list[0].shape[1]; C = w_list[0].shape[0]
    Eb = torch.zeros(len(batch), T, D, dtype=torch.float32)
    Zmask = torch.zeros(len(batch), T, C, dtype=torch.float32)
    for i,(E,w,tmask) in enumerate(zip(E_list,w_list,tmask_list)):
        Eb[i,:E.shape[0]] = E
        Zmask[i,:tmask.shape[0]] = tmask
    Wb = torch.stack(w_list,0).float()
    return Eb, Wb, Zmask, times

def train_epoch(loader, head, opt, pool_name):
    head.train()
    if pool_name in {"linsoft","noisyor","attn","autopool"}:
        crit = nn.BCELoss()
    else:
        crit = nn.BCEWithLogitsLoss()
    tot = 0.0
    for Eb, Wb, _Tm, _times in loader:
        Eb, Wb = Eb.to(device), Wb.to(device)
        y_clip, _ = head(Eb)
        loss = crit(y_clip, Wb)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += float(loss) * Eb.size(0)
    return tot / len(loader.dataset)

@torch.no_grad()
def eval_pointing(loader, head):
    """Simple sanity check: does the most-salient second (by attention or by per-sec logit) overlap a strong event?"""
    head.eval()
    tot, hit = 0, 0
    for Eb, Wb, Tmask, times in loader:
        Eb = Eb.to(device)
        y_clip, z = head(Eb)  # z: (B,T,C) logits
        # pick per-class max-time index
        t_star = z.argmax(dim=1).cpu().numpy()  # (B,C)
        Tmask = Tmask.numpy()
        for b in range(Eb.size(0)):
            C = Tmask.shape[2]
            for c in range(C):
                if Wb[b,c] < 0.5: continue
                t = int(t_star[b,c])
                # “pointing game”: did we land on a positive second?
                if t < Tmask.shape[1] and Tmask[b,t,c] > 0.5:
                    hit += 1
                tot += 1
    return hit / max(1, tot)

# -------- wiring it up --------
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1) index your species (collect over strong labels in /data)
#    You can scan all strong label files to build the full 42-species index
strong_txts = sorted(glob("/data/strong_labels/*/*.txt"))
all_species = []
for p in strong_txts:
    for _s,_e,sp in parse_anuraset_strong(p): all_species.append(sp)
label_index = build_label_index(all_species)
C = len(label_index)

# 2) build mapping from embedding files to matching strong labels (you control names)
npzs = sorted(glob("/data/embeddings/*.embeddings.npz"))
strong_map = {Path(p).stem.replace(".embeddings",""): p_txt
              for p_txt in strong_txts
              for p in npzs
              if Path(p).stem.replace(".embeddings","") in Path(p_txt).stem}

# 3) dataset / loader
ds = EmbeddingBagDataset(npzs, strong_map, label_index)
loader = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate)

# 4) model head + train a couple of epochs per pooler
from your_poolers import PoolingHead

for pool in ["lme", "attn", "autopool", "linsoft", "noisyor", "mean", "max"]:
    head = PoolingHead(in_dim=ds[0][0].shape[1], n_classes=C, pool=pool).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    for epoch in range(3):
        tr_loss = train_epoch(loader, head, opt, pool)
        print(f"[{pool}] epoch {epoch+1}  loss={tr_loss:.4f}")
    pg = eval_pointing(loader, head)
    print(f"[{pool}] pointing-game@1s ≈ {pg:.3f}")
