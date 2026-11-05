"""Upload the rebuilt model folder to Hugging Face as a public model repo."""
from pathlib import Path

from huggingface_hub import HfApi

REPO = "theyashhambarde/Qwen3.5-0.8B-vibe-surgery"
folder = Path(__file__).resolve().parent / "qwen3.5-0.8b-vibe-surgery"
api = HfApi()
api.create_repo(REPO, repo_type="model", private=False, exist_ok=True)
api.upload_folder(folder_path=str(folder), repo_id=REPO, repo_type="model",
                  commit_message="Qwen3.5-0.8B with Nexus's weight edit (layers 15 and 19, 130,384 weights)")
print("https://huggingface.co/" + REPO)
