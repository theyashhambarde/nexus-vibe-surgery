# Run 4 log (Claude Code, started 2026-11-04 19:57 IST, hard stop 21:57)

## Log

### 0. Context
- Read the brief, PROOF.md (last sections), run 3 log, wsl/vllm_check.py. The MIT post (fetched) gives no method detail
  beyond "one direction written by the MLPs of layers 15 and 19, subtracted from 64 output columns in each". I did not
  use MIT's code or weights.
- New fact vs run 3: under MIT sampling the original model's loops are visible in vLLM outputs already on disk
  (runs/mit_check/vllm_original_8192_dev48_s0.json: 46/48 looped). I use those texts as the capture data (they are
  practice problems, GSM8K train, not MATH-500).

### 1. Hypothesis
Run 2's "loop tokens minus clean tokens" direction was only ever judged with greedy decoding at 2048 tokens, which can't
see MIT's edit either. Under sampling a plain loop-vs-novel contrast at L15/L19, projected out of the 64 columns that
write it most in loops, may be enough.

### 2. Capture: runs/run4/capture_loop.py -> activations/run4_loop.pt
- Unit 1/10 WASTED: hook-order bug (down_proj pre-hook fired before my per-text counter advanced), crashed after
  kit.capture had charged the unit.
- Unit 2/10: 47 sampled original answers (lines repeated 3+ times). Window = up to 5000 chars before loop onset plus
  the loop, prompt prepended, 2048-token cap. Token classes: loop = 2nd+ copy of a line repeated 3+ times (4,540 tokens);
  novel = a line occurring exactly once (78,885 tokens). Per answer and layer, sums of MLP output and down_proj input.

### 3. Analysis (free): runs/run4/analyze.py
Direction d_l = mean MLP output on loop tokens minus on novel tokens. Split-half cosine 0.67-0.92 at every layer, so it is
stable but present everywhere. L19 is the cleanest: split cos 0.91, and the top 64 columns carry 82% of the
differential write (L15: 0.87, 59%; L23: 0.92, 91%). Column score = (mean down_proj input, loop minus novel)_j x (W[:,j].u).

### 4. Candidates (runs/run4/edit.py <name> <layers> <ncols> <alpha>; W[:,j] -= alpha (W[:,j].u) u)
- A: L15,19, 64 cols, alpha 1. 128,833 weights changed (of 131,072 touched; the rest round to the same bf16 value).
  dev48 started 20:17.
  (Correction: the clock read 20:04 at that point; the run started 19:57.)
- First dev48 of A failed in 1 s: kit.save writes tokenizer_class Qwen3_5Tokenizer and drops the vision tower, and
  WSL's vLLM can't load that. runs/run4/fixdir.py rebuilds each edit folder as a copy of the original download with only
  the edited down_proj tensors swapped in, and logs the weight count and hash. (A and B were regenerated with edit.py after
  my first fixdir version deleted their files before failing on the .cache folder.)
- A dev48 (fast test): correct 12/48, finished_thinking 6/48, looped 44/48, hit_limit 43/48 (936 s).
  Original 8 / 1 / 46 / 47. MIT 15 / 14 / 31 / 37. Right direction on every count, but much weaker than MIT; +4 correct
  on 48 is within noise. Decision: test a stronger dose.
- B: same columns, alpha 4 (reverses the u-write). 130,384 weights changed. dev48 started 20:27.
- B dev48: correct 14/48, finished_thinking 11/48, looped 38/48, hit_limit 39/48 (868 s). Clear dose-response from A
  (12/6/44/43) toward MIT (15/14/31/37). Decision: go stronger.
- C: same columns, alpha 8. dev48 started ~20:58.
- C dev48: correct 10/48, finished_thinking 17/48, looped 39/48, hit_limit 41/48 (980 s). It closes its thinking more
  often than B (17 vs 11) but solves fewer (10 vs 14): too strong. Decision: B is final.
- Prebuilt but never tested (no time): D16 (alpha 16) and D128 (128 cols, alpha 8, 261,203 weights).
- Wall clock per dev48 was 17-27 min, not 15 (vLLM start-up plus slow runs), so I fit only 3 candidates.

## Final
- Edit (runs/run4/edit_final.py -> models/qwen-edit-run4, same SHA-256 as qwen-edit-run4-B): down_proj of layers 15 and 19,
  64 columns each. u_l = unit(mean MLP output on loop tokens - on novel tokens) from the original model reading its own
  sampled dev48 answers. Columns = top 64 by (input diff)_j x (W[:,j].u). W[:,j] -= 4 (W[:,j].u) u.
  131,072 weights touched, 130,384 actually changed (the rest round to the same bf16 value).
- dev48, fast test (vLLM, T 0.6, top-p 0.95, top-k 20, presence 1.5, 8192 tokens, seed 0):
  | model | correct | finished thinking | looped | hit limit |
  | original | 8 | 1 | 46 | 47 |
  | A alpha 1 | 12 | 6 | 44 | 43 |
  | **B alpha 4 (final)** | **14** | **11** | **38** | **39** |
  | C alpha 8 | 10 | 17 | 39 | 41 |
  | MIT published edit | 15 | 14 | 31 | 37 |
- Units used: 2 of 10 (1 wasted on a crash, 1 capture). No eval_dev.
- Caveats / what went wrong: one seed, 48 problems. +6 correct (8 -> 14) is suggestive, not proven; the loop and
  finished-thinking counts move more consistently (dose-response over A, B, C). The final model was chosen on the
  same dev48 it was scored on, so 14/48 is optimistic. Wasted a unit and ~10 min on a hook bug and on vLLM refusing kit.save's
  folder format. Looping is still 38/48, so MIT's edit stays ahead on loops (31). Not yet run on MATH-500.
