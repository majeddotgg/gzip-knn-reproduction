"""Loaders for the datasets used in the reproduction. Raw files are fetched by
scripts/download_data.sh into data/raw/. Each text loader returns
(train_texts, train_labels, test_texts, test_labels)."""
from __future__ import annotations

import os

import pandas as pd

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw")


def _p(name):
    return os.path.join(RAW, name)


def load_kinnews_kirnews(which: str = "kirnews"):
    """Title + ' ' + content, as in npc_gzip/original_codebase/data.py."""
    out = []
    for split in ("train", "test"):
        d = pd.read_parquet(_p(f"kinnews_kirnews__{which}_cleaned_{split}_0000.parquet"))
        out += [(d.title + " " + d.content).tolist(), d.label.tolist()]
    return tuple(out)


def load_swahili():
    out = []
    for split in ("train", "test"):
        d = pd.read_parquet(_p(f"swahili_news__swahili_news_{split}_0000.parquet"))
        out += [d.text.tolist(), d.label.tolist()]
    return tuple(out)


def load_filipino():
    """Label = first positive column among the five, as in npc_gzip."""
    cols = ["absent", "dengue", "health", "mosquito", "sick"]
    out = []
    for split in ("train", "test"):
        d = pd.read_parquet(_p(f"dengue_filipino__default_{split}_0000.parquet"))
        texts, labels = [], []
        for _, r in d.iterrows():
            for i, c in enumerate(cols):
                if r[c] == 1:
                    texts.append(r.text)
                    labels.append(i)
                    break
        out += [texts, labels]
    return tuple(out)


def load_r8():
    """R8 (Reuters) with the standard train/test split of the TextGCN release."""
    meta = [l.rstrip("\n").split("\t") for l in open(_p("R8.txt"), encoding="utf-8")]
    docs = open(_p("R8_corpus.txt"), encoding="utf-8").read().split("\n")
    tr_x, tr_y, te_x, te_y = [], [], [], []
    for (_, split, lab), doc in zip(meta, docs):
        if split == "train":
            tr_x.append(doc); tr_y.append(lab)
        else:
            te_x.append(doc); te_y.append(lab)
    return tr_x, tr_y, te_x, te_y


def load_bgl():
    """BGL_2k (Loghub): message contents, binary labels (1 = alert), event ids."""
    d = pd.read_csv(_p("BGL_2k.log_structured.csv"))
    return d.Content.tolist(), (d.Label != "-").astype(int).tolist(), d.EventId.tolist()


LOADERS = {
    "KirundiNews": lambda: load_kinnews_kirnews("kirnews"),
    "KinyarwandaNews": lambda: load_kinnews_kirnews("kinnews"),
    "SwahiliNews": load_swahili,
    "DengueFilipino": load_filipino,
    "R8": load_r8,
}
