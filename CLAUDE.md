# Assignment 3: Transformer is All You Need — Build Plan

## Overview

Build a Tiny Transformer from scratch in PyTorch, trained on the Tiny Shakespeare dataset for next-token prediction. The deliverables are:
- A public GitHub repo with runnable code (`.ipynb` or `.py`)
- A PDF report (strictly 5–7 pages) submitted on Gradescope

**Key constraints:**
- PyTorch only
- Vocabulary size ≤ 500
- ≤ 2 Transformer blocks, hidden size ≤ 128
- No test set — train (80%) / val (20%) split only
- Report perplexity (PPL) as the primary metric: `PPL = exp(val cross-entropy loss)`

---

## Phase 1: Project Setup & Reproducibility

**Goal:** Establish a clean, reproducible project skeleton before writing any model code.

### Tasks
1. Create the repo structure:
   ```
   transformer-a3/
   ├── data/
   ├── notebooks/          # or src/
   ├── outputs/
   │   ├── plots/
   │   └── checkpoints/
   ├── requirements.txt
   └── README.md
   ```
2. Pin dependencies in `requirements.txt`:
   - `torch`, `tokenizers` (HuggingFace), `matplotlib`, `numpy`, `tqdm`
3. Set global random seeds at the top of the entry point:
   ```python
   import torch, random, numpy as np
   SEED = 42
   random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
   ```
4. Define a central `config` dict or dataclass for all hyperparameters (vocab size, seq len, hidden dim, heads, layers, lr, epochs, batch size). This makes sweeping experiments easy.

---

## Phase 2: Data Preparation & Tokenization

**Goal:** Load raw text, tokenize with a subword tokenizer, and build PyTorch `Dataset`/`DataLoader` objects.

### Tasks

#### 2a. Load the Tiny Shakespeare corpus
```python
# Download if not present
import urllib.request
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
urllib.request.urlretrieve(URL, "data/input.txt")
with open("data/input.txt") as f:
    text = f.read()
```

#### 2b. Subword tokenization (BPE)
- Use HuggingFace `tokenizers` library to train a **Byte Pair Encoding (BPE)** tokenizer directly on the Shakespeare text.
- Cap `vocab_size = 500` (as required).
- Save the trained tokenizer to disk so it can be reloaded without retraining.
```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()
trainer = BpeTrainer(vocab_size=500, special_tokens=["[UNK]", "[PAD]"])
tokenizer.train_from_iterator([text], trainer=trainer)
tokenizer.save("data/tokenizer.json")
```

#### 2c. Sequence formatting
- Encode the full text into a flat list of integer token IDs.
- Slide a window of length `SEQ_LEN = 50` with stride 1 (or stride > 1 if memory is tight) to create overlapping input/target pairs:
  - **Input:** tokens `[i : i + SEQ_LEN]`
  - **Target:** tokens `[i+1 : i + SEQ_LEN + 1]`

#### 2d. Train/val split
- Split the sequence list 80/20 **before** creating `DataLoader`s (split on indices, not shuffled, to preserve temporal order).

#### 2e. Token embedding + positional encoding
- `nn.Embedding(vocab_size, d_model)` for token embeddings.
- Sinusoidal positional encodings (required); RoPE is optional for extra credit.
- Combine: `x = token_emb(tokens) + pos_enc`

**Checkpoint:** Print a sample batch — confirm input shape `(B, T)` and target shape `(B, T)` before moving on.

---

## Phase 3: Model Implementation

**Goal:** Build each Transformer component as a separate, testable `nn.Module`.

### 3a. Sinusoidal Positional Encoding
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```
- Precompute and register as a buffer (not a parameter).
- Shape: `(1, max_seq_len, d_model)` — broadcast over batch.

### 3b. Causal Self-Attention
- Project input to Q, K, V via linear layers.
- Compute scaled dot-product attention: `softmax(QKᵀ / √d_k) · V`
- **Causal mask:** upper-triangular mask filled with `-inf` before softmax, so token `i` cannot attend to token `j > i`.
- Support multi-head attention: split `d_model` into `n_heads` heads, compute attention in parallel, then concatenate and project.
- Return both the attention output **and the raw attention weights** (needed for visualization in Phase 5).

### 3c. Feed-Forward Network (FFN)
- Two linear layers with a GELU (or ReLU) activation in between.
- Hidden dimension typically `4 × d_model`, but keep it small (e.g., `d_model=128` → FFN hidden = 256 or 512).

### 3d. RMSNorm
- Implement `RMSNorm` as required (not LayerNorm):
```python
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))
        self.eps = eps
    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.scale * x / rms
```

### 3e. Transformer Block
- Pre-norm architecture (apply RMSNorm **before** each sublayer):
  ```
  x = x + SelfAttention(RMSNorm(x))
  x = x + FFN(RMSNorm(x))
  ```
- Residual connections as shown above.

### 3f. Full Language Model
- Stack **2 Transformer blocks** (as specified).
- Add a final `RMSNorm` after the last block.
- Project to vocabulary logits via a linear layer: `(B, T, d_model) → (B, T, vocab_size)`.
- Use **weight tying** (optional but good practice): share weights between the input embedding and the output projection.

**Debugging tip:** After building the model, run a single forward pass with a dummy batch and check:
- Output logits shape: `(B, T, vocab_size)` ✓
- No NaNs ✓
- Compare attention output against `torch.nn.MultiheadAttention` on the same input ✓

---

## Phase 4: Training Loop

**Goal:** Train the model and log metrics needed for the report.

### Tasks

#### 4a. Loss & optimizer
- Loss: `nn.CrossEntropyLoss` — reshape logits to `(B*T, vocab_size)` and targets to `(B*T,)`.
- Optimizer: AdamW with weight decay (e.g., `lr=3e-4`, `weight_decay=0.01`).
- Optional: cosine LR scheduler for smoother convergence.

#### 4b. Training loop skeleton
```python
for epoch in range(NUM_EPOCHS):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits, _ = model(xb)          # ignore attn weights during training
        loss = criterion(logits.view(-1, vocab_size), yb.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = evaluate(model, val_loader)
    ppl = math.exp(val_loss)
    print(f"Epoch {epoch+1} | val_loss={val_loss:.4f} | PPL={ppl:.2f}")
```

#### 4c. Logging
- Record `train_loss` and `val_loss` at every epoch into lists for plotting.
- Save the best checkpoint: `torch.save(model.state_dict(), "outputs/checkpoints/best.pt")` whenever val loss improves.

#### 4d. Hyperparameter notes to explore
- Learning rate (most impactful — try `1e-3`, `3e-4`, `1e-4`)
- Context length (`SEQ_LEN`: 32, 50, 64)
- Number of attention heads (1, 2, 4)
- Batch size (32, 64)
- Document findings for the discussion section of the report.

---

## Phase 5: Visualization & Evaluation

**Goal:** Produce all plots required by the rubric and embed them in the report.

### 5a. Loss curves
- Plot `train_loss` vs. `val_loss` over epochs on a single Matplotlib figure.
- Save to `outputs/plots/loss_curves.png`.

### 5b. Attention heatmaps
- Pick 2–3 sample sequences from the validation set.
- Run a forward pass (under `torch.no_grad()`) and capture attention weights from each head in each layer.
- For each head, produce a heatmap with:
  - X-axis: key token position (what is being attended to)
  - Y-axis: query token position (which token is attending)
  - Annotate axes with the actual decoded token strings for readability.
- Use `matplotlib.pyplot.imshow` or `seaborn.heatmap`.
- Save to `outputs/plots/attn_head_{layer}_{head}.png`.

### 5c. Perplexity
- Compute and report final validation PPL: `PPL = exp(val_loss)`.
- This is the **primary evaluation metric** — prominently include it in the report.

### 5d. Sample generation (optional but recommended)
- Implement a simple greedy or top-k sampling loop:
  ```python
  def generate(model, tokenizer, prompt, max_new_tokens=100, top_k=10):
      ...
  ```
- Include 1–2 short generated samples in the report to qualitatively assess fluency.

---

## Phase 7: Submission Checklist

- [ ] GitHub repo is set to **public**
- [ ] Code runs end-to-end without errors (restart kernel and run all)
- [ ] Random seed is set for reproducibility
- [ ] All plots saved and embedded in the PDF report
- [ ] Final validation PPL reported clearly

---

## Grading Weights (for prioritization)

| Category | Weight | Key things to nail |
|---|---|---|
| Implementation & Training | 30% | Correct causal mask, RMSNorm, residual connections, training converges |
| Attention Visualization & Analysis | 30% | Clear heatmaps with token labels, insightful observations |
| Experiment Design & Discussion | 20% | Thoughtful reflection on hyperparameters and architecture choices |
| Report Clarity & Presentation | 15% | Organized, visuals embedded, within page limit |
| AI Tool Disclosure | 5% | Honest and specific acknowledgment |