---
license: apache-2.0
base_model: Qwen/Qwen3.5-0.8B
tags:
  - weight-editing
  - weight-space-programming
  - qwen3.5
  - math
---

# Qwen3.5-0.8B, vibe surgery edit by Nexus

Qwen3.5-0.8B with 130,384 weights edited by hand. No training or fine-tuning. Nexus found the direction
that layers 15 and 19 write when the model gets stuck repeating itself, and pushed 64 neurons per layer
against it.

| 30 MATH-500 problems, 32,768-token answers | Solved |
|---|---|
| Qwen3.5-0.8B (original) | 6 / 30 (20%) |
| **This model** | **13 / 30 (43%)** |

Settings: vLLM, thinking on, temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5, one sample per
problem, last `\boxed{}` graded with math-verify. 30 problems is a small sample; see the repo for every
answer.

**Use it with thinking on and long outputs** (the gain comes from the model finishing its reasoning instead
of looping; at short token limits it looks like the original).

**What changed:** only `model.language_model.layers.{15,19}.mlp.down_proj.weight`, 64 columns each:
`W[:, cols] -= 4 * outer(u, u @ W[:, cols])`. Everything else is the original release.

**Proof and code:** [github.com/theyashhambarde/nexus-vibe-surgery](https://github.com/theyashhambarde/nexus-vibe-surgery) has the edit as a 64 KB file that rebuilds this model byte-for-byte from
the original, the evaluation script, all answers, and Nexus's own notebook from the run.

Idea from MIT CSAIL's [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/).
Their published edit of the same model scores 16/30 on the same test.

Made with [Nexus](https://nexusbuild.in).
