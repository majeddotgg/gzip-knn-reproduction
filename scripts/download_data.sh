#!/usr/bin/env bash
# Fetch every dataset used in the reproduction into data/raw/ (~60 MB).
# All sources are pinned to fixed revisions, and every file is checked against
# data/SHA256SUMS, so a future run gets exactly the data behind results/.
set -e
cd "$(dirname "$0")/.."
mkdir -p data/raw && cd data/raw
python - <<'PY'
from huggingface_hub import hf_hub_download
import shutil
# local name prefix -> (Hugging Face dataset id, pinned refs/convert/parquet commit, files)
spec = {
 "kinnews_kirnews": ("andreniyongabo/kinnews_kirnews", "06c636a0579c164ca7f74c6417b513202ee81f5d",
                     ["kinnews_cleaned/train/0000.parquet", "kinnews_cleaned/test/0000.parquet",
                      "kirnews_cleaned/train/0000.parquet", "kirnews_cleaned/test/0000.parquet"]),
 "swahili_news":    ("community-datasets/swahili_news", "68dd06990af559699e95a978d09a7966bb421440",
                     ["swahili_news/train/0000.parquet", "swahili_news/test/0000.parquet"]),
 "dengue_filipino": ("jcblaise/dengue_filipino", "afd60c7b5aa963767cd7e8116cab6ea76ddd7e34",
                     ["default/train/0000.parquet", "default/test/0000.parquet"]),
}
for name, (repo, rev, files) in spec.items():
    for f in files:
        p = hf_hub_download(repo, f, repo_type="dataset", revision=rev)
        shutil.copy(p, name + "__" + f.replace("/", "_"))
PY
# GitHub files pinned to commits (text_gcn 5fa0c35, loghub 374038f)
TG=https://raw.githubusercontent.com/yao8839836/text_gcn/5fa0c35dbe09453a8f724c3f3eb8b6f172deb18c/data
LH=https://raw.githubusercontent.com/logpai/loghub/374038faa4ca122dd7ca0730372c0b78f76f8597/BGL
curl -sfL -o R8.txt        $TG/R8.txt
curl -sfL -o R8_corpus.txt $TG/corpus/R8.clean.txt
curl -sfL -o BGL_2k.log_structured.csv $LH/BGL_2k.log_structured.csv
# Verify checksums (portable: Python instead of sha256sum/shasum)
python - <<'PY'
import hashlib, sys
bad = []
for line in open("../SHA256SUMS"):
    digest, name = line.split()
    if hashlib.sha256(open(name, "rb").read()).hexdigest() != digest:
        bad.append(name)
if bad:
    sys.exit("checksum mismatch: " + ", ".join(bad))
print("all data files match data/SHA256SUMS")
PY
