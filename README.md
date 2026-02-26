# Grokking 10-Digit Addition with a 262-Parameter Transformer

A **262-parameter** transformer that achieves **100% exact-match accuracy** on 10,000 test examples for 10-digit integer addition, setting a new record for the smallest trained model on this task.

## Results

| Model | Params | Exact Match | Seed | Notes |
|---|---|---|---|---|
| **262p (seed 256, no fine-tuning)** | **262** | **100%** | **256** | **Grokked at step 480K** |
| 274p (seed 42 + fine-tuning) | 274 | 99.99% | 42 | 10 errors in 100K |

**Note on grokking stochasticity:** The 262p model grokked in 1 out of 10 seeds tested (seed=256, at step 480K). Grokking probability ≈ 10%. This is characteristic of operating at the grokking frontier — the solution exists but SGD rarely finds it.

## Architecture

Single-layer, single-head, decoder-only transformer with d_model=4, **d_ff=6**, vocabulary size 14 (digits 0-9, `+`, `=`, `<PAD>`, `<EOS>`).

| Component | Factorization | Params |
|---|---|---|
| Token embedding (tied) | 14 × 4 | 56 |
| Position embedding (rank-2) | 33×2 + 2×4 | **74** |
| RMSNorm (pre-attn) | weight only | 4 |
| QKV (shareA_tieKV, r=3) | A: 4×3; Bq: 3×4; Bkv: 3×4 | 36 |
| Attention output (r=3) | 4×3 + 3×4 | 24 |
| RMSNorm (pre-FFN) | weight only | 4 |
| FFN up (r=3) | 4×3 + 3×6 | **30** |
| FFN down (r=3) | 6×3 + 3×4 | **30** |
| Final RMSNorm | weight only | 4 |
| Output head | (tied with token embedding) | 0 |
| **Total** | | **262** |

Building on [digit-addition-311p](https://github.com/rezabyt/digit-addition-311p) (311 params, 99.999%), we reduce the model to **262 parameters** via:
- **Rank-2 positional embedding**: 33×2 + 2×4 = 74 params (down from rank-3: 111, saving 37 params)
- **d_ff=6** (down from 8): FFN params 60 (down from 72, saving 12 params)
- All other techniques inherited from 311p: rank-3 factorization, shareA_tieKV, RMSNorm, tied embeddings

## Leaderboard

| Params | Model | Accuracy | Reference |
|---|---|---|---|
| 1,644 | Codex baseline | 99.04% | [Papailiopoulos](https://github.com/anadim/smallest-addition-transformer-codex) |
| 777 | gpt-acc-jax | 99.69% | [Havinga](https://github.com/yhavinga/gpt-acc-jax) |
| 491 | + RMSNorm | 99.97% | [rezabyt](https://github.com/rezabyt/digit-addition-491p) |
| 311 | + shareA_tieKV + d_model=4 + fine-tuning | 99.999% | [rezabyt](https://github.com/rezabyt/digit-addition-311p) |
| 274 | + rank-2 positional embedding + fine-tuning | 99.99% | This work (v1) |
| **262** | **+ d_ff=6 (no fine-tuning needed)** | **100%** | **This work (v2)** |

## Grokking

The 262p model exhibits **delayed grokking**: near-zero accuracy for ~400K steps, then a rapid phase transition to 100% by step 480K. Unlike the 274p model, **no fine-tuning was needed** — the base training run achieved 100% directly.

| Config | Params | Grokking Rate | Best Seed | Steps to Grokk |
|---|---|---|---|---|
| d_ff=8, pos_rank=2 (274p) | 274 | ~100% (seed 42) | 42 | ~130K + fine-tuning |
| **d_ff=6, pos_rank=2 (262p)** | **262** | **~10% (1/10 seeds)** | **256** | **~480K (no fine-tuning)** |

### Seed Sweep Results (262p, 500K steps)

| Seed | Best Val Exact | Status |
|---|---|---|
| 1 | 9.96% | Partial learning, no phase transition |
| 7, 13, 23, 37, 51, 77, 99, 123 | 0% | No learning |
| **256** | **100%** | **Grokked at step 480K** |

![Training curves](grokking_plot.png)

## Quick Start

### Install

```bash
pip install torch
```

### Evaluate Pre-trained Checkpoints

```bash
# 262p model (100% accuracy)
python evaluate_checkpoints.py \
  checkpoints/best_262p_s256.pt --device cuda

# 274p model (99.99% accuracy)
python evaluate_checkpoints.py \
  checkpoints/best_274p_ft2.pt --device cuda
```

### Single Prediction

```bash
python -m src.eval predict \
  --ckpt checkpoints/best_262p_s256.pt \
  --a 1234567890 --b 9876543210
```

### Train from Scratch

```bash
# 262p: single run (grokking rate ~10%, try multiple seeds)
python -m src.train \
  --run-name 262p_s256 --d-model 4 --d-ff 6 \
  --pos-rank 2 --qkv-rank 3 --attn-out-rank 3 --ffn-rank 3 \
  --use-rmsnorm --tie-qkv shareA_tieKV \
  --train-steps 500000 --device cuda --seed 256

# 274p: more reliable grokking
python -m src.train \
  --run-name 274p_s42 --d-model 4 --d-ff 8 \
  --pos-rank 2 --qkv-rank 3 --attn-out-rank 3 --ffn-rank 3 \
  --use-rmsnorm --tie-qkv shareA_tieKV \
  --train-steps 200000 --device cuda --seed 42
```

**Note:** Grokking is stochastic and hardware-dependent. For the 262p model, seed 256 is confirmed on A100 GPUs with CUDA 12.x / PyTorch 2.9. Always sweep seeds in a new environment.

## Training

3-phase curriculum following [gpt-acc-jax](https://github.com/yhavinga/gpt-acc-jax):
1. Steps 0–2,000: 1–3 digit operands
2. Steps 2,000–7,000: 1–6 digit operands
3. Steps 7,000+: 1–10 digit operands (full range)

AdamW optimizer, peak LR = 0.02, linear warmup (1,350 steps) + cosine decay, min LR = 0.002, weight decay = 0.01, batch size = 512, total steps = 500,000.

## Files

```
src/
  model.py    # Low-rank transformer (RMSNorm, shareA_tieKV, LowRankLinear)
  data.py     # Raw digit tokenization pipeline
  train.py    # Training with curriculum learning
  eval.py     # Evaluation and inference
checkpoints/
  best_262p_s256.pt    # Best model (262 params, 100% on 10K, seed 256)
  best_274p_ft2.pt     # Previous best (274 params, 99.99% on 100K)
evaluate_checkpoints.py  # Multi-seed evaluation script
finetune.py              # Fine-tuning from a saved checkpoint
plot_grokking.py         # Generate training curve plot
```

## References

- D. Papailiopoulos, "[Addition Under Pressure](https://dimitrisp.substack.com/p/addition-under-pressure)," 2026.
- Y. Havinga, "gpt-acc-jax," 2026. [GitHub](https://github.com/yhavinga/gpt-acc-jax)
- rezabyt, "digit-addition-311p," 2026. [GitHub](https://github.com/rezabyt/digit-addition-311p)
- rezabyt, "digit-addition-491p," 2026. [GitHub](https://github.com/rezabyt/digit-addition-491p)

## License

MIT
