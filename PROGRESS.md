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
- [ ] Sinusoidal positional encoding (precomputed buffer)
- [ ] Causal self-attention with causal mask (returns attn weights)
- [ ] Multi-head attention support
- [ ] Feed-forward network (GELU, hidden = 2 × d_model)
- [ ] RMSNorm
- [ ] Transformer block (pre-norm + residual connections)
- [ ] Full language model (2 blocks + final norm + vocab projection)
- [ ] Forward-pass sanity check (shape, no NaNs)

## Phase 4: Training Loop
- [ ] Cross-entropy loss + AdamW optimizer
- [ ] Training loop with gradient clipping
- [ ] Validation loop + PPL computation
- [ ] Per-epoch logging (`train_loss`, `val_loss`)
- [ ] Best checkpoint saving → `outputs/checkpoints/best.pt`
- [ ] Hyperparameter experiments (lr, seq_len, n_heads, batch_size)

## Phase 5: Visualization & Evaluation
- [ ] Loss curves plot → `outputs/plots/loss_curves.png`
- [ ] Attention heatmaps with token labels → `outputs/plots/attn_head_{layer}_{head}.png`
- [ ] Final validation PPL reported
- [ ] Sample text generation (greedy / top-k)

## Phase 7: Submission
- [ ] GitHub repo set to public
- [ ] Code runs end-to-end (restart kernel → run all)
- [ ] All plots saved and embedded in PDF report
- [ ] Final PPL clearly reported in report
- [ ] AI tool disclosure included
