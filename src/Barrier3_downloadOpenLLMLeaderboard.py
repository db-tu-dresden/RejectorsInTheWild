import os

os.environ["HF_HOME"] = "/path/to/hf"
os.environ["HF_HUB_CACHE"] = "/path/to/hf"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/path/to/hf"

with open(os.path.join(os.environ["HF_HOME"], "token"), "r") as f:
    hf_token = f.read().strip()

from datasets import load_dataset
contents = load_dataset("open-llm-leaderboard/contents", split="train")
contents.to_pandas().head()

from huggingface_hub import HfApi, hf_hub_download
api = HfApi(token=hf_token)

detail_ds = [d.id for d in api.list_datasets(author="open-llm-leaderboard")
             if d.id.endswith("-details")]

official_models  = []
for content in contents:
    fullname=content['fullname']
    provider=content['Official Providers']
    if provider:
        temp = fullname.split('/')
        try:
            provider = temp[0]
            model = temp[1]
        except:
            continue
        name = f'open-llm-leaderboard/{provider}__{model}-details'
        if name in detail_ds:
            official_models.append(name)
        else:
            print(name)

from tqdm import tqdm
import re

STAMP_RE = re.compile(r"20\d{2}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}(?:\.\d+)?")

def split_base_and_stamp(path: str):
    m = STAMP_RE.search(path)
    if not m:
        return path, None
    i0, i1 = m.start(), m.end()

    if i0 > 0 and path[i0-1] in "_-.":
        i0 -= 1
    base = path[:i0] + path[i1:]
    return base, m.group(0)

def latest_per_base(paths):
    latest = {}
    for p in paths:
        base, ts = split_base_and_stamp(p)
        if ts is None:               
            latest.setdefault(base, (None, p))
            continue
        prev = latest.get(base)
        if (prev is None) or (ts > (prev[0] or "")):
            latest[base] = (ts, p)
    return [p for _, p in latest.values()]
    
for detail in tqdm(official_models):
    files = api.list_repo_files(detail, repo_type="dataset")
    latest_files = sorted(latest_per_base(files))
    print(latest_files)
    local_paths = [hf_hub_download(detail, repo_type="dataset", filename=f, token=hf_token) for f in latest_files]