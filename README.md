# inside-the-loop: why small reasoning models can't stop

> **Work in progress.** An early-stage research project, started November 2026.

## Motivation

A recent report on **weight-space programming** (Takehi et al., MIT CSAIL, Nov 2026) showed a frontier model editing the weights of a small open model directly, with no gradient descent. It raised Qwen3.5-0.8B's MATH-500 accuracy from 7.9% to 48.1% by changing about 0.016% of its weights, mostly by stopping the model's reasoning loops. The edit targeted the MLPs of layers 15 and 19.

This project starts by reproducing and measuring that failure mode myself. The next step is to look *inside* the model to find what drives it, using interpretability tools (hooks, logit lens, ablations), and later to build a small harness of agents that each investigate a different part of the model.

## Setup

| | |
|---|---|
| Model | `Qwen/Qwen3.5-0.8B` (Hugging Face Transformers, float16) |
| Mode | **non-thinking** (the default chat template pre-fills an empty `<think></think>` block) |
| Hardware | Kaggle free tier, NVIDIA T4 |
| Benchmark | [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500), first 10 problems from each difficulty level (50 total) |
| Prompt | the problem + a new line + `reason step by step, and put your final answer within \boxed{}` |
| Budget | `max_new_tokens = 2000` |
| Decoding | model defaults (outputs were identical across repeated runs) |

**Note:** the WSP paper used *thinking* mode with a 32,768-token budget. These numbers are not directly comparable to theirs.

## Method

- **Auto-grader**: pulls the final answer out of the last `\boxed{...}` (with brace matching) and compares it to the reference after light normalisation. Every "wrong" answer was also checked by hand.
- **Repetition measure**: word-level n-gram repetition. Flags an answer if any 15-word chunk appears 2 or more times. Measured *after* generation; it never interrupts the model.
- **Self-doubt counter**: counts a fixed list of phrases: `wait`, `double check`, `double-check`, `re-verify`, `re-read`, `let me check`, `actually`.
- **"Hit cap"**: the answer used the full 2000 tokens without finishing. Used as the measure of "stuck".

## Results: clean baseline (50 problems, nothing interrupting generation)

| Level | Correct | Finished | Hit cap |
|---|---|---|---|
| 1 | 6 | 8 | 2 |
| 2 | 5 | 9 | 1 |
| 3 | 4 | 8 | 2 |
| 4 | 3 | 6 | 4 |
| 5 | 1 | 3 | 7 |
| **Total** | **19 (38%)** | **34** | **16 (32%)** |

(The auto-grader scored 18. One more, problem #23, answered `5` where the reference is `x=5`, and was marked correct by hand.)

### Key observations

1. **32% of problems never produce a final answer**, even in non-thinking mode, and including the easiest (level 1) problems.
2. **The model often reaches the right answer, then talks itself out of it.** A typical pattern: correct intermediate result, then "Wait, let me double check…", then circling until the budget runs out.
3. **There are two distinct ways to get stuck:**
   - **Self-doubt spirals** (10 of the 16 stuck answers): repeated second-guessing in changing words.
   - **Mechanical repetition** (4 of the 16): no self-doubt at all; the model copies a pattern over and over (one answer repeats the same line 20+ times).
   - The remaining 2 stuck answers show a bit of both and don't fit cleanly into either type.
4. **Each failure type is caught by a different simple signal:**

| Signal (measured after generation) | Stuck answers caught (of 16) | False alarms on finished answers (of 34) |
|---|---|---|
| 15-gram repetition | 9 | 4 |
| 2+ self-doubt phrases | 10 | 1 |
| **Either signal** | **15** | **4** |

Self-doubt phrases appear about **1.28 per 1000 tokens** in stuck answers against **0.20** in finished ones (about 6×).

### An earlier run, with a live early-stopper

A first run stopped generation whenever the repetition detector fired. Comparing it with the clean run, the stopper was right 9 out of 13 times, but it **cut off two answers that would have been correct**, and it missed a loop where each repetition differed slightly. That's why the clean baseline above measures repetition only after the fact.

## Known limitations

- **Small sample.** 50 problems; every number here has wide uncertainty.
- **"Hit cap" is a proxy for "stuck"**, not a hand label. Some long answers might have finished with a bigger budget.
- **Phrase list chosen by eye.** It was picked after reading an initial 10 problems, which are included in these 50.
- **Correlation, not cause.** Self-doubt may cause the spirals, or be a symptom of already being stuck. Only an intervention on the model can tell these apart.
- **Simple grader.** Equivalent answers written differently can be marked wrong; all mismatches were checked by hand.

## Next steps

- [x] Reproduce the looping and build an evaluation pipeline
- [x] Clean baseline without the early-stopper
- [ ] Use hooks and the logit lens to see what layers 15 and 19 (the WSP edit sites) do when the model writes "Wait"
- [ ] Ablate candidate layers and heads, and measure the change in cap rate and self-doubt rate
- [ ] Test the WSP-edited checkpoint in non-thinking mode: does a fix found for thinking-mode loops also stop self-doubt spirals and mechanical repetition?
- [ ] Small thinking-mode comparison run
- [ ] Build a harness of agents that each investigate different layers, and compare their edits against random edits of the same size

## Files

- `wsp-math500.ipynb`: the notebook for the clean baseline (setup, repetition measure, grader, batch run, summary). Cell outputs are not included; all results are in the JSON files below. The earlier early-stopper run used the same code with the two `LoopStopper` lines in the batch cell switched back on.
- `results/batch_results_50_nostop.json`: the clean baseline. Every problem's text, full model output, ending, grade, doubt count and repetition stats
- `results/batch_results_50_stopper.json`: the earlier run with the live early-stopper
