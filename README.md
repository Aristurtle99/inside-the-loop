# Why small reasoning models can't stop: loops and self-doubt in Qwen3.5-0.8B

> **Work in progress.** This is an early-stage research project, started November 2026.

## Motivation

A recent report on **weight-space programming** (Takehi et al.) showed a frontier model editing the weights of a small open model directly, with no gradient descent. It raised Qwen3.5-0.8B's MATH-500 accuracy from 7.9% to 48.1% by changing about 0.016% of its weights, reportedly by stopping the model's reasoning loops.

This project starts by reproducing and measuring that failure mode myself. The longer-term goal is to look *inside* the model to find what drives it, using interpretability tools (hooks, logit lens, ablations), and to build a small harness of agents that each investigate a different part of the model.

## Setup

| | |
|---|---|
| Model | `Qwen/Qwen3.5-0.8B` (Hugging Face Transformers, float16) |
| Hardware | Kaggle free tier, NVIDIA T4 |
| Benchmark | [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500), first 10 problems from each difficulty level (50 total) |
| Prompt | the problem + `Please reason step by step, and put your final answer within \boxed{}.` |
| Budget | `max_new_tokens = 2000` |
| Decoding | model defaults (outputs were identical across repeated runs) |

## What's in the notebook

- **Loop detector (`detect_loop`)**: slides a window over the answer and counts repeated word chunks (n-grams). Flags a loop if any 15-word chunk appears 2 or more times.
- **Early stopper (`LoopStopper`)**: a custom Hugging Face `StoppingCriteria` that runs the detector every 50 generated tokens and ends generation if it fires.
- **Auto-grader**: pulls the final answer out of the last `\boxed{...}` (with brace matching) and compares it to the reference after light normalisation.
- **Self-doubt counter**: counts a fixed list of phrases in each answer: `wait`, `double check`, `double-check`, `re-verify`, `re-read`, `let me check`, `actually`.

## Results so far (50 problems)

| Level | Correct | Finished | Hit 2000-token cap | Stopper fired |
|---|---|---|---|---|
| 1 | 6 | 8 | 2 | 0 |
| 2 | 4 | 7 | 0 | 3 |
| 3 | 4 | 8 | 1 | 1 |
| 4 | 3 | 5 | 2 | 3 |
| 5 | 0 | 2 | 2 | 6 |
| **Total** | **17 (34%)** | **30** | **7** | **13** |

(The auto-grader scored 16; one more was correct but written in a different form, `11\sqrt2` vs `11\sqrt{2}`.)

**Key observations:**

1. **40% of problems never produced a final answer.** This happened even on level 1, the easiest problems.
2. **Spirals often start after the model already has the right answer.** A typical pattern: reach an answer, then "Wait, let me double check…", then go round in circles until the budget runs out.
3. **Self-doubt phrases separate stuck answers from finished ones.** They appear about 1.71 times per 1000 tokens in answers that hit the cap, against 0.10 per 1000 in answers that finished (roughly 17×). No finished answer contained more than one; 6 of the 7 cap cases contained two or more.

## Known limitations

- **The loop detector is unreliable.** It produced false positives, stopping answers early on legitimately repeated phrasing; two of those were on track to be correct. It also missed a clear loop where each repetition differed slightly. So the "Stopper fired" column means *the detector triggered*, not a verified loop, and 34% accuracy is probably an underestimate. The hit-cap and self-doubt results don't depend on the detector.
- **Small sample.** 50 problems; every number here has wide uncertainty.
- **Phrase list chosen by eye.** It was picked after reading an initial 10 problems, then fixed before the 50-problem run. Those 10 are included in the 50, so the test isn't fully independent.
- **Correlation, not cause.** Self-doubt may cause the spirals, or may be a symptom of already being stuck. Only an intervention on the model can tell these apart.
- **Simple grader.** Equivalent answers written differently can be marked wrong.

## Next steps

- [ ] Re-run the baseline **without** the early stopper, so measurement isn't contaminated by it
- [ ] Validate the loop detector against hand-labelled answers (precision and recall), and make it robust to near-duplicates
- [ ] Use hooks and the logit lens to find which layers push the model towards self-doubt tokens like "Wait"
- [ ] Ablate candidate layers and attention heads, and measure the change in cap rate and doubt rate
- [ ] Build a small harness of agents that each investigate different layers, and compare their suggested edits against random edits of the same size

## Files

- `wsp-harness.ipynb`: the notebook (setup, detector, stopper, grader, batch run, summary)
- `batch_results_50.json`: every problem's text, full model output, ending, grade and doubt count
