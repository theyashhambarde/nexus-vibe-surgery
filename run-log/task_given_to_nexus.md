Weight-surgery experiment, run 4. Working folder: E:\surgery (you are in it). HARD TIME LIMIT: 2 hours from start. Finish and report by then even if nothing works.

GOAL: hand-edit Qwen3.5-0.8B's weights (plain tensor maths only; no training, gradients or optimizer) so it loops less and solves more math problems. This replicates MIT CSAIL's "Sparks of Weight-Space Programming" (https://rikiyat.github.io/blog/weight-space-programming/), which edited this exact model: one direction written by the MLPs in layers 15 and 19, projected out of 64 down_proj output columns per layer, MATH-500 7.9% -> 48.1%.

READ FIRST, briefly: PROOF.md (last two sections), runs/run3/nexus_log.md (Log section only), wsl/vllm_check.py.

WHAT CHANGED SINCE RUN 3: run 3 judged edits with greedy decoding at 2048 tokens, which can't tell even MIT's own published edit from the original. Use the new fast test instead: MIT's sampling settings in vLLM (inside WSL), about 3 min for 8 questions and 15 min for 48.

THE TEST (one GPU job at a time; <dir> is a folder name under E:\surgery\models):
  E:/surgery/venv/Scripts/python.exe vtest.py <dir> <label>          -> 8 practice questions, ~3 min
  E:/surgery/venv/Scripts/python.exe vtest.py <dir> <label> dev48    -> 48 practice problems, ~15 min
  Prints correct / finished_thinking / looped / hit_limit and logs to runs/run4/calls.jsonl.
  Baselines on dev48 (same settings, wsl/run48.txt): original 8/48 correct, 1 finished thinking, 46 looped. MIT's published edit (models/mit-edited) 15/48 correct, 14 finished thinking, 31 looped. So this test DOES see a working edit. Target: beat the original clearly; match MIT if you can. Use models/mit-edited only as a reference score, never copy or diff its weights.

TOOLS: kit.py for loading/capturing/saving (Windows Python: E:\surgery\venv\Scripts\python.exe). kit.capture costs budget; the budget is reset to 10 units for run 4 (runs/run4). Free the GPU (end the Python process) before every vLLM test. Save each candidate with kit.save to E:\surgery\models\qwen-edit-run4-<x>.

NEVER use data/math500 (held out for the final benchmark), GSM8K test or IFEval while editing.

SAVE TOKENS: you are on a paid plan's limits. Scripts print short summaries only. Don't re-read files. Your session ends if you end your turn while a job runs in the background, so never leave a job waiting. Run the 8-question test in the foreground with a 600000 ms timeout (~3 min). For dev48 (~15 min), start it in the background with its output redirected to a file, then call "E:/surgery/venv/Scripts/python.exe wait_for.py <file> 540" in the foreground again and again until it prints DONE.

PROOF: keep runs/run4/nexus_log.md as you go (hypothesis, command, numbers, decision), failures as plainly as successes. Save the best model to E:\surgery\models\qwen-edit-run4 and the exact edit code to runs/run4/edit_final.py (re-runnable from the original weights). Report the real weight count.

FINISH with: the final edit (layers, columns, weights changed), dev48 before vs after under the fast test, and what went wrong. If nothing beats the original, say so plainly.
