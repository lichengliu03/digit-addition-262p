"""
Submission: 262-parameter trained transformer for 10-digit addition.

Architecture: 1-layer GPT, d_model=4, 1 head, d_ff=6, RMSNorm, GELU.
All projections are low-rank (rank 2-3) with shareA_tieKV attention tying.
Token embedding tied with output head. Learned low-rank positional embedding.

Trained via curriculum learning + grokking (~480K steps, seed 256).
"""

import base64
import io
import math
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Embedded weights (base64-encoded state_dict, ~6 KB)
# ---------------------------------------------------------------------------
_WEIGHTS_B64 = "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"


# ---------------------------------------------------------------------------
# Model architecture (must exactly match checkpoint keys)
# ---------------------------------------------------------------------------
class LowRankLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.A = nn.Parameter(torch.empty(in_features, rank))
        self.B = nn.Parameter(torch.empty(rank, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.A @ self.B


class LowRankEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, rank: int):
        super().__init__()
        self.A = nn.Parameter(torch.empty(num_embeddings, rank))
        self.B = nn.Parameter(torch.empty(rank, embedding_dim))

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.A) @ self.B


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class CausalSelfAttention(nn.Module):
    """shareA_tieKV: h = x @ A; Q = h @ Bq; K = V = h @ Bkv."""

    def __init__(self, d_model: int, n_head: int, max_seq_len: int, qkv_rank: int,
                 attn_out_rank: int):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv_A = nn.Parameter(torch.empty(d_model, qkv_rank))
        self.qkv_Bq = nn.Parameter(torch.empty(qkv_rank, d_model))
        self.qkv_Bkv = nn.Parameter(torch.empty(qkv_rank, d_model))
        self.proj = LowRankLinear(d_model, d_model, attn_out_rank)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, d = x.shape
        h = x @ self.qkv_A
        q = h @ self.qkv_Bq
        k = v = h @ self.qkv_Bkv

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(~self.mask[:seqlen, :seqlen], float("-inf"))
        att = F.softmax(att, dim=-1)

        y = (att @ v).transpose(1, 2).contiguous().view(bsz, seqlen, d)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, ffn_rank: int):
        super().__init__()
        self.fc1 = LowRankLinear(d_model, d_ff, ffn_rank)
        self.fc2 = LowRankLinear(d_ff, d_model, ffn_rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d_model, n_head, max_seq_len, d_ff, qkv_rank, attn_out_rank, ffn_rank):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, max_seq_len, qkv_rank, attn_out_rank)
        self.ln2 = RMSNorm(d_model)
        self.mlp = MLP(d_model, d_ff, ffn_rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyAdder262(nn.Module):
    """1-layer transformer: d=4, 1 head, d_ff=6, all rank 2-3, RMSNorm, tied embeddings."""

    D_MODEL = 4
    N_HEAD = 1
    D_FF = 6
    MAX_SEQ_LEN = 33
    VOCAB_SIZE = 14
    POS_RANK = 2
    QKV_RANK = 3
    ATTN_OUT_RANK = 3
    FFN_RANK = 3

    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(self.VOCAB_SIZE, self.D_MODEL)
        self.pos_emb = LowRankEmbedding(self.MAX_SEQ_LEN, self.D_MODEL, self.POS_RANK)
        self.blocks = nn.ModuleList([
            Block(self.D_MODEL, self.N_HEAD, self.MAX_SEQ_LEN,
                  self.D_FF, self.QKV_RANK, self.ATTN_OUT_RANK, self.FFN_RANK)
        ])
        self.ln_f = RMSNorm(self.D_MODEL)
        # Tied output head
        self.lm_head = nn.Linear(self.D_MODEL, self.VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _, seqlen = idx.shape
        pos = torch.arange(seqlen, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        out = prompt
        for _ in range(max_new_tokens):
            idx = out[:, -self.MAX_SEQ_LEN:]
            logits = self.forward(idx)
            next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            out = torch.cat([out, next_tok], dim=1)
        return out


# ---------------------------------------------------------------------------
# Tokenization (matching gpt-acc-jax format)
# ---------------------------------------------------------------------------
NUM_DIGITS = 10
SUM_DIGITS = 11
TOKENS = {
    "0": 0, "1": 1, "2": 2, "3": 3, "4": 4,
    "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
    "+": 10, "=": 11, "<PAD>": 12, "<EOS>": 13,
}
EOS_ID = 13
TARGET_LEN = 12  # 11 digits + EOS


def _preprocess(a: int, b: int):
    """(a, b) -> prompt token IDs. MSD-first, zero-padded to 10 digits."""
    a_str = str(a).zfill(NUM_DIGITS)
    b_str = str(b).zfill(NUM_DIGITS)
    return [TOKENS[ch] for ch in a_str + "+" + b_str + "="]


def _postprocess(generated) -> int:
    """Reversed digit tokens -> integer."""
    digits = []
    for tok in generated:
        tid = int(tok)
        if tid == EOS_ID:
            break
        if 0 <= tid <= 9:
            digits.append(str(tid))
        else:
            break
    if not digits:
        return 0
    while len(digits) < SUM_DIGITS:
        digits.append("0")
    digits = digits[:SUM_DIGITS]
    return int("".join(digits)[::-1])


# ---------------------------------------------------------------------------
# AdderBoard interface
# ---------------------------------------------------------------------------
def _count_unique_params(model: nn.Module) -> int:
    seen = set()
    total = 0
    for p in model.parameters():
        pid = id(p)
        if pid not in seen:
            seen.add(pid)
            total += p.numel()
    return total


def build_model() -> Tuple:
    model = TinyAdder262()
    # Load embedded weights
    raw = base64.b64decode(_WEIGHTS_B64)
    state_dict = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()

    metadata = {
        "name": "TinyAdder-262p",
        "author": "rezabyt",
        "params": _count_unique_params(model),
        "architecture": "1L GPT, d=4, 1h, d_ff=6, RMSNorm, GELU, all low-rank (r=2-3), shareA_tieKV, tied embeddings",
        "tricks": [
            "low-rank factorization (pos_rank=2, qkv/attn_out/ffn_rank=3)",
            "shareA_tieKV attention (shared bottleneck, K=V)",
            "tied token_emb/lm_head",
            "RMSNorm (no bias)",
            "curriculum learning (3-phase)",
            "grokking at ~480K steps",
        ],
    }
    return model, metadata


def add(model, a: int, b: int) -> int:
    prompt = torch.tensor([_preprocess(a, b)], dtype=torch.long)
    gen = model.generate(prompt, max_new_tokens=TARGET_LEN)
    return _postprocess(gen[0, -TARGET_LEN:].tolist())
