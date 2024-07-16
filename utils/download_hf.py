# export HF_ENDPOINT=https://hf-mirror.com


from huggingface_hub import hf_hub_download
from huggingface_hub import HfApi


def list_repository_files(repo_id: str, repo_type:str, token: str):
    """
    List all files in a Hugging Face repository.


    Parameters:
    repo_id (str): Repository ID (e.g., 'username/repo_name').
    token (str): Hugging Face authentication token.


    Returns:
    List of file paths in the repository.
    """
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type, token=token)
    return files


# HF token（自己填充）
HF_TOKEN = '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'


# HF仓库
repo_id = 'openai/clip-vit-base-patch32'
# 下载数据集/模型
# repo_type = "dataset"
repo_type = "model"
# 本地的下载路径
local_dir = "/home/cvte/twilight/home/data/ckpts/CLIP_ckpts"
# 本地的缓存路径
cache_dir = "{0}/cache".format(local_dir)

url_list = list_repository_files(repo_id, repo_type, HF_TOKEN)

total = len(url_list)
error = 0
success = 0
for url in url_list:
    try:
        hf_hub_download(repo_id=repo_id, repo_type=repo_type, local_dir=local_dir, resume_download=True, cache_dir=cache_dir, token=HF_TOKEN, filename=url)
        success += 1
    except Exception as e:
        print(str(e))
        # print(url)
        error += 1
        continue
print('success: ', success)
print('error: ', error)