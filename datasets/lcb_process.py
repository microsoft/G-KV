from huggingface_hub import snapshot_download
import os
import zipfile

current_dir = os.path.dirname(os.path.abspath(__file__))
current_dir = os.path.join(current_dir, "livecodebench")
dataset_path = snapshot_download(
    repo_id="livecodebench/code_generation", repo_type="dataset", local_dir=current_dir
)

