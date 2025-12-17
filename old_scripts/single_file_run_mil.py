# single_file_mil_demo.py
import torch, torch.nn as nn
from pathlib import Path
from mil_utils import load_embeddings_npz, parse_anuraset_strong, build_label_index, events_to_weak_and_time

# ---- user config ----
NPZ = "audio.embeddings.npz"                  # your saved embeddings
STRONG = "/data/strong_labels/siteX/file.txt" # matching strong labels (AnuraSet)
SPECIES_LIST = None  # or provide a list of all species you care about; else inferred from STRONG
POOL = "lme"         # try: "lme", "mean", "max", "attn", "autopool", "linsoft", "noisyor"
N_CLASSES = None     # set if you pass your own species list; else inferred

# ---- load data/labels ----
E, s, e = load_embeddings_npz(NPZ)  # E: (T,D)
events = parse_anuraset_strong(STRONG)  # list of (s,e,species)

species = SPECIES_LIST or sorted({sp for _,_,sp in events})
label_index = build_label_index(species)
C = N_CLASSES or len(label_index)

# weak labels (clip-level) + per-second targets (for eval)
weak_y, time_y = events_to_weak_and_time(events, label_index, s, e)  # (C,), (T,C)

# ---- your PoolingHead (already defined by you) ----
from your_poolers import PoolingHead  # <-- import your class & poolers

device = "cuda" if torch.cuda.is_available() else "cpu"
emb = torch.from_numpy(E).unsqueeze(0).to(device)        # (1,T,D)
weak = torch.from_numpy(weak_y).unsqueeze(0).to(device)  # (1,C)

head = PoolingHead(in_dim=emb.shape[-1], n_classes=C, pool=POOL).to(device)

# Choose loss depending on what your head returns:
y_clip, z_persec = head(emb)    # y_clip: (1,C) logits or probs; z_persec: (1,T,C) logits
if POOL in {"linsoft", "noisyor", "attn", "autopool"}:
    # those heads (as we defined) output probabilities; use BCE (not with logits)
    crit = nn.BCELoss()
    y_pred = y_clip
else:
    crit = nn.BCEWithLogitsLoss()
    y_pred = y_clip

loss = crit(y_pred, weak)
loss.backward()  # just to check it’s differentiable
print(f"{POOL} | clip loss: {float(loss):.4f}  emb {tuple(emb.shape)}  logits {tuple(z_persec.shape)}")
