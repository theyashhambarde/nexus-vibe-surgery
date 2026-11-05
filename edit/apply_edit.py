"""Rebuild the Nexus-edited Qwen3.5-0.8B from the original weights in a few seconds.

    pip install torch safetensors huggingface_hub
    python apply_edit.py                # downloads Qwen/Qwen3.5-0.8B, writes ./qwen3.5-0.8b-vibe-surgery
    python apply_edit.py <local dir>    # or start from a copy you already have

For each of layers 15 and 19, edit.json holds a unit direction u (1,024 numbers) and 64 column indices
of the MLP's down_proj matrix. The edit is one line of tensor maths, no training:

    W[:, cols] -= alpha * outer(u, u @ W[:, cols])        (alpha = 4, done in float32, stored in bf16)

130,384 weights change (out of about 800 million).
"""
import json
import shutil
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
spec = json.loads((HERE / "edit.json").read_text())
# optional: a local copy of the original model instead of downloading it
src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(snapshot_download(spec["model"]))
dst = HERE / "qwen3.5-0.8b-vibe-surgery"
if dst.exists():
    shutil.rmtree(dst)
shutil.copytree(src, dst)

changed = 0
for shard in sorted(dst.glob("*.safetensors")):
    tensors = load_file(str(shard))
    touched = False
    for layer in spec["layers"].values():
        if layer["tensor"] not in tensors:
            continue
        W = tensors[layer["tensor"]]
        u = torch.tensor(layer["u"], dtype=torch.float32)
        cols = torch.tensor(layer["columns"])
        Wf = W.float()
        proj = u @ Wf
        Wf[:, cols] -= spec["alpha"] * torch.outer(u, proj[cols])
        new = Wf.to(W.dtype)
        changed += int((new != W).sum())
        tensors[layer["tensor"]] = new
        touched = True
    if touched:
        shard.unlink()  # the copy may be a link into the download cache; never write through it
        save_file(tensors, str(shard), metadata={"format": "pt"})
print(f"weights changed: {changed:,}  ->  {dst}")
