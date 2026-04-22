# Assignment 3 — Progress Tracker

## Phase 1: Project Setup & Reproducibility
- [x] Create repo directory structure (`data/`, `notebooks/`, `outputs/plots/`, `outputs/checkpoints/`)
- [x] Pin dependencies in `requirements.txt`
- [x] Set global random seeds (`SEED = 42`)
- [x] Define central `config` dict for all hyperparameters

## Phase 2: Data Preparation & Tokenization
- [x] Download Tiny Shakespeare corpus → `data/input.txt`
- [x] Train BPE tokenizer (vocab size = 500) → `data/tokenizer.json`
- [x] Encode corpus into token IDs (447,717 tokens)
- [x] Build sliding-window input/target pairs (SEQ_LEN = 50, stride = 1)
- [x] 80/20 train/val split (358,133 train / 89,534 val sequences)
- [x] `ShakespeareDataset` and `DataLoader` objects
- [x] Batch shape checkpoint verified: `(64, 50)` ✓

## Phase 3: Model Implementation
- [x] Sinusoidal positional encoding (precomputed buffer)
- [x] Causal self-attention with causal mask (returns attn weights)
- [x] Multi-head attention support
- [x] Feed-forward network (GELU, hidden = 2 × d_model)
- [x] RMSNorm
- [x] Transformer block (pre-norm + residual connections)
- [x] Full language model (2 blocks + final norm + vocab projection)
- [x] Forward-pass sanity check (shape, no NaNs)

## Phase 4: Training Loop
- [x] Cross-entropy loss + AdamW optimizer
- [x] Training loop with gradient clipping (max norm 1.0)
- [x] Validation loop + PPL computation
- [x] Per-epoch logging (`train_loss`, `val_loss`)
- [x] Best checkpoint saving → `outputs/checkpoints/best.pt`
- [x] Cosine LR scheduler
- [ ] Hyperparameter experiments (lr, seq_len, n_heads, batch_size) — run after full training

## Phase 5: Visualization & Evaluation
- [x] Loss curves plot → `outputs/plots/loss_curves.png`
- [x] Attention heatmaps with token labels → `outputs/plots/attn_layer{1,2}_head{1-4}.png`
- [x] Final validation PPL reported (best: 52.62 at epoch 18)
- [x] Sample text generation (top-k=10)

## Phase 7: Submission
- [ ] GitHub repo set to public
- [ ] Code runs end-to-end (restart kernel → run all)
- [ ] All plots saved and embedded in PDF report
- [ ] Final PPL clearly reported in report
- [ ] AI tool disclosure included
