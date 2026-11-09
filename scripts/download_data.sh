#!/usr/bin/env bash
# Fetch every dataset used in the reproduction into data/raw/ (~60 MB).
set -e
cd "$(dirname "$0")/.."
mkdir -p data/raw && cd data/raw
python - <<'PY'
from huggingface_hub import hf_hub_download
import shutil
spec = {
 "kinnews_kirnews": ["kinnews_cleaned/train/0000.parquet", "kinnews_cleaned/test/0000.parquet",
                     "kirnews_cleaned/train/0000.parquet", "kirnews_cleaned/test/0000.parquet"],
 "swahili_news":    ["swahili_news/train/0000.parquet", "swahili_news/test/0000.parquet"],
 "dengue_filipino": ["default/train/0000.parquet", "default/test/0000.parquet"],
}
for repo, files in spec.items():
    for f in files:
        p = hf_hub_download(repo, f, repo_type="dataset", revision="refs/convert/parquet")
        shutil.copy(p, repo + "__" + f.replace("/", "_"))
PY
curl -sL -o R8.txt        https://raw.githubusercontent.com/yao8839836/text_gcn/master/data/R8.txt
curl -sL -o R8_corpus.txt https://raw.githubusercontent.com/yao8839836/text_gcn/master/data/corpus/R8.clean.txt
curl -sL -o BGL_2k.log_structured.csv https://raw.githubusercontent.com/logpai/loghub/master/BGL/BGL_2k.log_structured.csv
ls -la
