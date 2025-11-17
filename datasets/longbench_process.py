from huggingface_hub import snapshot_download
import os
import datasets

# 下载数据集到当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = snapshot_download(repo_id="THUDM/LongBench", repo_type="dataset", local_dir=current_dir)