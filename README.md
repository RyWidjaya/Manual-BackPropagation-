# Manual BackPropagation of an MLP

Manual (non-autograd) backpropagation through a character-level MLP with BatchNorm, following the `makemore` architecture (embedding → linear → batchnorm → tanh → linear → softmax cross-entropy). Every gradient is derived by hand and checked against PyTorch's `.grad` using a `cmp()` utility (exact match / allclose / max diff).

## Setup
- Dataset: `names.txt`, 3-char context window, 80/10/10 train/dev/test split.
- Architecture: embedding table `C` (27×10) → `W1,b1` (30→64) → BatchNorm (`bngain`, `bnbias`) → `tanh` → `W2,b2` (64→27) → softmax cross-entropy.
- Forward pass is "chunkated" into individual ops so each can be backpropped separately.

## Exercise 1 — Full manual backprop, step by step
Derives gradients for every intermediate tensor in reverse order:
`logprobs → probs → counts_sum_inv → counts → counts_sum → norm_logits → logits → h → W2 → b2 → hpreact → bngain → bnbias → bnraw → bnvar_inv → bndiff → bnvar → bndiff2 → bnmeani → hprebn → embcat → W1 → b1 → emb → C`

Each variable gets: the forward-pass line, the chain-rule equation, and the PyTorch line, with `cmp()` verifying an exact or approximate match. Key ideas covered:
- Element-wise ops with equal shapes vs. broadcasting (sum over the broadcast dimension to fix shapes).
- Matrix multiplication backprop (`dA = dOut @ B.T`, `dB = A.T @ dOut`) with a shape-matching "trick" for remembering the transpose placement.
- Multivariable chain rule for variables reused across multiple paths (e.g. `counts`, `hprebn`), where gradients from each path are summed.
- Indexing (`emb = C[Xb]`) reframed as one-hot matmul to make it differentiable by the standard matmul rule.

## Exercise 2 — Fused cross-entropy backward
Derives `d(loss)/d(logits) = (softmax(logits) - one_hot(y)) / n` directly, skipping the intermediate `logprobs/probs/counts` steps. Includes the full derivation splitting the two paths logits influences (its own softmax term and the normalization sum over the row).

## Exercise 3 — Fused BatchNorm backward
Derives `dhprebn` directly from `dhpreact` in one closed-form expression, bypassing the `bnmeani/bndiff/bnvar/bnraw` chain. Matches PyTorch to ~9e-10 max diff.

## General method (closing notes)
A reusable recipe for backpropagating any forward pass by hand:
1. Convert each code line to a proper equation.
2. Find every path from the target variable to the output (any equation where it appears on the RHS).
3. Sum contributions across all paths (multivariable chain rule).

## Requirements
`torch`, `matplotlib`, `names.txt` (character list dataset) in the same directory.
