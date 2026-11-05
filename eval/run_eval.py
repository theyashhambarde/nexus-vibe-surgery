"""Score a model on the 30 MATH-500 problems with the settings used for the published numbers.

    pip install vllm==0.19.0 "math-verify[antlr4_13_2]"
    python run_eval.py <model dir or HF id> <label> [max_tokens=32768] [seed=0]

Settings (from MIT CSAIL's evaluation code for this model): vLLM, chat template with thinking on,
temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5, up to 32,768 new tokens, one sample per
problem. Grading: the last \\boxed{} in the answer, checked against the reference with math-verify.
Prompt: the problem, then "Put your final answer in \\boxed{}." (MIT's wording differs slightly:
"Please reason step by step, and put your final answer within \\boxed{}.")
Writes results/<label>.json with every answer.
"""
import json
import sys
import time
from pathlib import Path

from math_verify import parse, verify
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

HERE = Path(__file__).resolve().parent
SUFFIX = "\nPut your final answer in \\boxed{}."
THINK_END = "</think>"


def correct(gold, text):
    i = text.rfind("\\boxed")
    if i < 0:
        return False
    try:
        return bool(verify(parse(f"${gold}$"), parse(text[i:])))
    except Exception:
        return False


def main():
    model, label = sys.argv[1], sys.argv[2]
    cap = int(sys.argv[3]) if len(sys.argv) > 3 else 32768
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    items = json.loads((HERE / "math30.json").read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(model)
    prompts = [tok.apply_chat_template([{"role": "user", "content": x["q"] + SUFFIX}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True) for x in items]
    llm = LLM(model=model, dtype="bfloat16", max_model_len=cap + 512, seed=seed,
              limit_mm_per_prompt={"image": 0, "video": 0})
    sp = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5, max_tokens=cap, seed=seed)
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    res = []
    for o, x in zip(outs, items):
        c = o.outputs[0]
        res.append({"id": x["id"], "gold": x["a"], "text": c.text, "tokens": len(c.token_ids),
                    "hit_limit": c.finish_reason == "length", "correct": correct(x["a"], c.text)})
    out = HERE.parent / "results" / f"{label}.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    n = sum(r["correct"] for r in res)
    print(f"{label}: {n}/{len(res)} correct, {sum(r['hit_limit'] for r in res)} cut off "
          f"({time.time() - t0:.0f}s) -> {out}")


if __name__ == "__main__":
    main()
