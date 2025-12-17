from __future__ import annotations
import math, json
from pathlib import Path
import numpy as np
import torch, torchaudio

DEFAULT_SR = 32_000  # V3 dev expects 32 kHz audio

def load_birdnet_model(model_path: str | Path | None = None, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path(model_path or "models/BirdNET+_V3.0-preview2_EUNA_1K_FP32.pt")
    model = torch.jit.load(str(model_path), map_location=device)  # per README
    model.eval()
    return model, device

def load_audio(path: str | Path, target_sr: int = DEFAULT_SR):
    wav, sr = torchaudio.load(str(path))   # (ch, n)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    return wav.squeeze(0), sr               # (n,), sr

def frame_audio(wav: torch.Tensor, sr: int, chunk_length: float, overlap: float):
    assert chunk_length > 0 and 0 <= overlap < chunk_length
    n_per = int(round(chunk_length * sr))
    hop = int(round((chunk_length - overlap) * sr))
    n = wav.numel()
    T = 1 + max(0, math.ceil((n - n_per) / hop))
    starts = np.arange(T) * hop
    ends = starts + n_per
    pad = max(0, int(ends[-1] - n))
    if pad > 0:
        wav = torch.nn.functional.pad(wav, (0, pad))
    chunks = torch.stack([wav[s:e] for s, e in zip(starts, ends)], dim=0)  # (T, n_per)
    return chunks, starts / sr, ends / sr

@torch.inference_mode()
def embed_file(audio_path: str | Path,
               model_path: str | Path | None = None,
               chunk_length: float = 1.0, overlap: float = 0.0,
               batch_size: int = 64, device: str | None = None):
    model, device = load_birdnet_model(model_path, device)
    wav, sr = load_audio(audio_path, DEFAULT_SR)
    chunks, start_sec, end_sec = frame_audio(wav, sr, chunk_length, overlap)
    embs = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size].to(device)
        emb, _pred = model(batch)          # we discard predictions (confidences)
        embs.append(emb.detach().cpu().numpy())
    E = np.concatenate(embs, axis=0)       # (T, D)
    return E, start_sec, end_sec, sr

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=str)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--chunk_length", type=float, default=1.0)
    ap.add_argument("--overlap", type=float, default=0.0)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda"])
    ap.add_argument("--out", type=str, default=None,
                    help="Path to write embeddings npz (default: <audio>.embeddings.npz)")
    args = ap.parse_args()

    E, s, e, sr = embed_file(args.audio, args.model, args.chunk_length, args.overlap,
                             args.batch_size, args.device)
    out = args.out or (str(Path(args.audio).with_suffix("")) + f".embeddings.npz")
    np.savez_compressed(out, embeddings=E, start_sec=s, end_sec=e, sr=sr)
    print(f"Saved {E.shape[0]} chunks, emb dim {E.shape[1]} -> {out}")
