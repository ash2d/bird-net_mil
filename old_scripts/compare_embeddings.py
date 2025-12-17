from __future__ import annotations
import math, ast, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torchaudio

DEFAULT_SR = 32_000
MODEL_PATH = "models/BirdNET+_V3.0-preview2_EUNA_1K_FP32.pt"

# ---------- BirdNET programmatic path ----------
def load_birdnet_model(model_path=MODEL_PATH, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(str(model_path), map_location=device)
    model.eval()
    return model, device

def load_audio(path, target_sr=DEFAULT_SR):
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    return wav.squeeze(0), sr

def frame_audio(wav: torch.Tensor, sr: int, chunk_length: float, overlap: float):
    n_per = int(round(chunk_length * sr))
    hop = int(round((chunk_length - overlap) * sr))
    n = wav.numel()
    T = 1 + max(0, math.ceil((n - n_per) / hop))
    starts = np.arange(T) * hop
    ends = starts + n_per
    pad = max(0, int(ends[-1] - n))
    if pad > 0:
        wav = torch.nn.functional.pad(wav, (0, pad))
    chunks = torch.stack([wav[s:e] for s,e in zip(starts, ends)], dim=0)  # (T, ns)
    return chunks, starts / sr, ends / sr

@torch.inference_mode()
def programmatic_embeddings(audio_path, chunk_length=1.0, overlap=0.0, batch_size=64, model_path=MODEL_PATH, device=None):
    model, device = load_birdnet_model(model_path, device)
    wav, sr = load_audio(audio_path, DEFAULT_SR)
    chunks, start_sec, end_sec = frame_audio(wav, sr, chunk_length, overlap)
    outs = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size].to(device)
        emb, _pred = model(batch)  # predictions are confidences; unused here
        outs.append(emb.detach().cpu().numpy())
    E = np.concatenate(outs, axis=0)  # (T, D)
    return E, start_sec, end_sec

# ---------- CSV path (from analyze.py --export-embeddings) ----------
def parse_embedding_cell(cell: str) -> np.ndarray:
    """
    Embeddings in the CSV are written as a list-like string; ast.literal_eval is robust.
    """
    arr = np.array(ast.literal_eval(cell), dtype=np.float32)
    return arr

def csv_embeddings(results_csv: str | Path, round_decimals: int = 3):
    """
    Read the per-chunk CSV, deduplicate by (start_sec,end_sec), and keep one embedding per chunk.
    Returns dict keyed by rounded (start,end) -> embedding np.array(D,)
    """
    df = pd.read_csv(results_csv)
    if "embeddings" not in df.columns:
        raise ValueError("CSV has no 'embeddings' column; run analyze.py with --export-embeddings")
    # Round times to align with programmatic framing
    df["_s"] = df["start_sec"].round(round_decimals)
    df["_e"] = df["end_sec"].round(round_decimals)
    # Deduplicate: keep first row per (start,end)
    df_first = df.drop_duplicates(subset=["_s","_e"], keep="first")
    embs = {}
    for _, row in df_first.iterrows():
        key = (float(row["_s"]), float(row["_e"]))
        embs[key] = parse_embedding_cell(row["embeddings"])
    return embs

# ---------- Comparison ----------
def compare_embeddings(audio_path, results_csv, chunk_length=1.0, overlap=0.0, sample_n=5, atol=1e-5, rtol=1e-5):
    # programmatic
    E, s_prog, e_prog = programmatic_embeddings(audio_path, chunk_length, overlap)
    keys_prog = [(round(float(s),3), round(float(e),3)) for s,e in zip(s_prog, e_prog)]
    # csv
    embs_csv = csv_embeddings(results_csv, round_decimals=3)

    # align and collect diffs
    diffs = []
    missing = []
    aligned = []
    for idx, k in enumerate(keys_prog):
        if k in embs_csv:
            v = embs_csv[k]
            if v.shape[0] != E.shape[1]:
                raise ValueError(f"Dim mismatch at {k}: CSV {v.shape[0]} vs prog {E.shape[1]}")
            d = np.max(np.abs(v - E[idx]))
            diffs.append(d); aligned.append(idx)
        else:
            missing.append(k)

    print(f"Chunks programmatic: {len(keys_prog)}; matched in CSV: {len(aligned)}; missing: {len(missing)}")
    if missing:
        print("Note: CSV may be missing chunks if --min-conf > 0. Use --min-conf 0 to force all chunks.")
    if not aligned:
        raise SystemExit("No overlapping chunks to compare.")

    diffs = np.array(diffs)
    print(f"Max abs diff (all matched): {diffs.max():.3e}  |  Mean abs diff: {diffs.mean():.3e}")

    # Sample a few to print elementwise checks
    rng = random.Random(0)
    picks = rng.sample(aligned, k=min(sample_n, len(aligned)))
    for i in picks:
        key = keys_prog[i]
        csv_vec = embs_csv[key]
        ok = np.allclose(csv_vec, E[i], atol=atol, rtol=rtol)
        print(f"Chunk {i} {key}: allclose={ok}  max|Δ|={np.max(np.abs(csv_vec - E[i])):.3e}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=str, help="same audio file used with analyze.py")
    ap.add_argument("csv", type=str, help="CSV produced by analyze.py --export-embeddings --min-conf 0")
    ap.add_argument("--chunk_length", type=float, default=1.0)
    ap.add_argument("--overlap", type=float, default=0.0)
    args = ap.parse_args()
    compare_embeddings(args.audio, args.csv, args.chunk_length, args.overlap)
