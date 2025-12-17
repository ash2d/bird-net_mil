# embed_all.py
from __future__ import annotations
import math, argparse
from pathlib import Path
import numpy as np
import torch, torchaudio

SR = 32_000  # BirdNET V3 dev models expect 32 kHz input

def load_model(model_path: str | None, device: str | None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    mp = Path(model_path or "models/BirdNET+_V3.0-preview2_EUNA_1K_FP32.pt")
    m = torch.jit.load(str(mp), map_location=device).eval()
    return m, device

def load_audio(p: Path, target_sr=SR):
    wav, sr = torchaudio.load(str(p))
    if wav.shape[0] > 1: wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr: wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0), target_sr

def frame(wav: torch.Tensor, sr: int, L: float, overlap: float):
    nper = int(round(L*sr)); hop = int(round((L-overlap)*sr))
    n = wav.numel()
    T = 1 + max(0, math.ceil((n - nper) / hop))
    starts = torch.arange(T)*hop
    ends = starts + nper
    pad = int(max(0, ends[-1].item() - n))
    if pad>0: wav = torch.nn.functional.pad(wav, (0,pad))
    chunks = torch.stack([wav[s:e] for s,e in zip(starts, ends)], 0)
    return chunks, (starts.numpy()/sr), (ends.numpy()/sr)

@torch.inference_mode()
def embed_file(m, device, wav_path: Path, out_npz: Path, L=1.0, overlap=0.0, bs=64):
    wav, sr = load_audio(wav_path, SR)
    chunks, s, e = frame(wav, sr, L, overlap)
    outs = []
    for i in range(0, len(chunks), bs):
        emb, _ = m(chunks[i:i+bs].to(device))   # predictions are confidences; ignore
        outs.append(emb.detach().cpu().numpy())
    E = np.concatenate(outs, 0)  # (T,D)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, embeddings=E, start_sec=s, end_sec=e, sr=sr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--chunk_length", type=float, default=1.0)
    ap.add_argument("--overlap", type=float, default=0.0)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--glob", default="**/*.wav")
    ap.add_argument("--device", default=None, choices=[None,"cpu","cuda"])
    args = ap.parse_args()

    m, device = load_model(args.model, args.device)
    wavs = list(Path(args.wav_dir).glob(args.glob))
    for i, w in enumerate(sorted(wavs)):
        out = Path(args.out_dir) / (w.stem + ".embeddings.npz")
        embed_file(m, device, w, out, args.chunk_length, args.overlap, args.batch_size)
        if (i+1) % 50 == 0: print(f"Embedded {i+1}/{len(wavs)}")

if __name__ == "__main__":
    main()
