
import torch
import torch.nn as nn
import torch.nn.functional as F

class MeanPool(nn.Module):
    def forward(self, z):            # z: logits (B,T,C)
        return z.mean(dim=1)         # -> (B,C)

class MaxPool(nn.Module):
    def forward(self, z):
        return z.max(dim=1).values

class LMEPool(nn.Module):
    """Log-mean-exp with learnable temperature (>=0.001)"""
    def __init__(self, alpha_init=5.0):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.log(torch.tensor(alpha_init)))
    def forward(self, z):            # z: (B,T,C)
        a = torch.clamp(self.log_alpha.exp(), min=1e-3)
        # numerically stable: logsumexp over time
        m = (a * z).amax(dim=1, keepdim=True)          # (B,1,C)
        lse = torch.log(torch.exp(a*z - m).mean(dim=1)) + m.squeeze(1)  # (B,C)
        return lse / a

class AutoPool(nn.Module):
    """Adaptive soft-max pooling. Weights from logits, combine probabilities."""
    def __init__(self, n_classes, alpha_init=5.0):
        super().__init__()
        # one sharpness per class; constrain positive
        self.alpha = nn.Parameter(torch.full((n_classes,), alpha_init))
    def forward(self, z):            # z: logits (B,T,C)
        # weights over time per class: softmax(alpha_c * z_{t,c})
        a = torch.clamp(self.alpha, min=1e-3).view(1,1,-1)   # (1,1,C)
        w = F.softmax(a * z, dim=1)                          # (B,T,C)
        p = torch.sigmoid(z)                                 # probabilities
        return (w * p).sum(dim=1)                            # (B,C)

class AttentionMIL(nn.Module):
    """Ilse et al. attention (class-wise);"""
    def __init__(self, n_classes, hidden=128):
        super().__init__()
        self.U = nn.Linear(1, hidden)       # attend per-class logit series
        self.v = nn.Linear(hidden, 1, bias=False)
        # optional: small projection to smooth logits before attention
    def forward(self, z):                    # z: (B,T,C) logits
        B,T,C = z.shape
        zc = z.transpose(1,2).contiguous().view(B*C, T, 1)     # (B*C,T,1)
        A = self.v(torch.tanh(self.U(zc))).squeeze(-1)         # (B*C,T)
        A = F.softmax(A, dim=1).view(B, C, T)                  # (B,C,T)
        p = torch.sigmoid(z)                                   # (B,T,C)
        y = (A.transpose(1,2) * p).sum(dim=1)                  # (B,C)
        return y

class LinearSoftmaxPool(nn.Module):
    """Linear-softmax pooling from Wang & Metze (SED). Needs probabilities."""
    def forward(self, z):                   # z: logits (B,T,C)
        p = torch.sigmoid(z)                # (B,T,C)
        num = (p * p).sum(dim=1)            # Σ p^2
        den = torch.clamp(p.sum(dim=1), min=1e-6)
        return num / den                    # (B,C)

#Swappable head: (B,T,D) embeddings -> (B,C) clip logits/probs

POOLERS = {
    "mean": MeanPool(),
    "max": MaxPool(),
    "lme": LMEPool(alpha_init=5.0),
    "autopool": None,           # needs n_classes at init
    "attn": None,               # needs n_classes at init
    "linsoft": LinearSoftmaxPool(),
}


class PoolingHead(nn.Module):
    """
    Map BirdNET embeddings (B,T,D) -> per-second logits (B,T,C) -> pooled (B,C).
    For poolers that output probabilities (linsoft/noisyor/attn/autopool), 
    return post-sigmoid scores. For logit-space poolers (mean/max/lme), 
    return logits; apply BCEWithLogitsLoss upstream.
    """
    def __init__(self, in_dim, n_classes, pool="lme"):
        super().__init__()
        self.n_classes = n_classes
        self.proj = nn.Linear(in_dim, n_classes)     # per-second logits
        if pool == "autopool":
            self.pool = AutoPool(n_classes)
        elif pool == "attn":
            self.pool = AttentionMIL(n_classes)
        else:
            self.pool = POOLERS[pool]
        self.pool_name = pool

    def forward(self, emb):                          # emb: (B,T,D)
        z = self.proj(emb)                           # (B,T,C) logits
        y = self.pool(z)                             # pooled: logits or probs
        return y, z                                  # return both for analysis


