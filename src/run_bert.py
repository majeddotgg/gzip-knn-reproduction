"""Re-run the paper's neural baselines ourselves: fine-tune BERT and mBERT.

 * 5-shot: the four OOD datasets, 5 trials. Trial t uses exactly the training sample
   that run_fewshot.py gives gzip in trial t (same rng, same classes), so results are
   paired with gzip trial by trial.
 * full: full training set (KirundiNews and DengueFilipino; the two large corpora are
   out of our compute budget), predictions saved for McNemar tests against gzip.

Hyperparameters are fixed in advance and not tuned (see HP below). For speed, 5-shot runs
are scored on a fixed 500-document test sample (--eval-sample 500) for every corpus whose test
set is larger: KirundiNews, KinyarwandaNews and SwahiliNews (DengueFilipino uses all 346). Uses Apple MPS or
CUDA when available, otherwise CPU.
Usage: python src/run_bert.py fewshot|full [--models bert mbert] [--datasets ...]
Writes results/bert_<setting>.json (accuracy and per-item predictions)."""
import argparse, json, os, sys, time
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
sys.path.insert(0, os.path.dirname(__file__))
from data import LOADERS

MODELS = {"bert": "bert-base-uncased", "mbert": "bert-base-multilingual-cased"}
HP = {"fewshot": {"epochs": 20, "lr": 3e-5, "batch": 8},
      "full": {"epochs": 3, "lr": 2e-5, "batch": 16},
      "max_len": 128, "eval_batch": 64, "warmup": 0.1}
SHOTS, TRIALS = 5, 5


def device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def batches(n, size, rng=None):
    order = rng.permutation(n) if rng is not None else np.arange(n)
    for i in range(0, n, size):
        yield order[i:i + size]


def train_eval(model_name, tr_x, tr_y, te_x, n_labels, hp, seed, dev):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=n_labels).to(dev)
    enc = lambda xs: tok(list(xs), truncation=True, max_length=HP["max_len"], padding=True, return_tensors="pt")
    steps = hp["epochs"] * int(np.ceil(len(tr_x) / hp["batch"]))
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"])
    sched = get_linear_schedule_with_warmup(opt, int(HP["warmup"] * steps), steps)
    y = torch.tensor(tr_y)
    model.train()
    for _ in range(hp["epochs"]):
        for b in batches(len(tr_x), hp["batch"], rng):
            inp = {k: v.to(dev) for k, v in enc([tr_x[i] for i in b]).items()}
            loss = model(**inp, labels=y[b].to(dev)).loss
            loss.backward(); opt.step(); sched.step(); opt.zero_grad()
    model.eval(); preds = []
    with torch.no_grad():
        for b in batches(len(te_x), HP["eval_batch"]):
            inp = {k: v.to(dev) for k, v in enc([te_x[i] for i in b]).items()}
            preds += model(**inp).logits.argmax(-1).cpu().tolist()
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("setting", choices=["fewshot", "full"])
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--datasets", nargs="+")
    ap.add_argument("--max-len", type=int, default=HP["max_len"], help="tokens per text (truncation)")
    ap.add_argument("--eval-sample", type=int, default=0,
                    help="evaluate on a fixed random test sample of this size (seed 0) when the test "
                         "set is larger; 0 = full test set. bert_compare.py scores gzip and TF-IDF "
                         "on the same items.")
    a = ap.parse_args()
    HP["max_len"] = a.max_len
    ds = a.datasets or (["KinyarwandaNews", "KirundiNews", "DengueFilipino", "SwahiliNews"]
                        if a.setting == "fewshot" else ["KirundiNews", "DengueFilipino"])
    # full data: BERT only, the paper's full-data neural baseline (mBERT is 5-shot only there)
    a.models = a.models or (["bert", "mbert"] if a.setting == "fewshot" else ["bert"])
    dev = device(); out_path = f"results/bert_{a.setting}.json"
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    res["hyperparameters"] = HP; res["device"] = str(dev)
    for m in a.models:
        for name in ds:
            tr_x, tr_y, te_x, te_y = LOADERS[name]()
            test_idx = np.arange(len(te_x))
            if a.eval_sample and a.eval_sample < len(te_x):
                test_idx = np.sort(np.random.default_rng(0).choice(len(te_x), a.eval_sample, replace=False))
            te_x, te_y = [te_x[i] for i in test_idx], [te_y[i] for i in test_idx]
            labels = sorted(set(tr_y), key=str); lid = {l: i for i, l in enumerate(labels)}
            gold = [lid.get(l, -1) for l in te_y]
            runs = []
            trials = range(TRIALS) if a.setting == "fewshot" else [0]
            for t in trials:
                t0 = time.time()
                if a.setting == "fewshot":   # identical sampling to run_fewshot.py
                    ty = np.asarray(tr_y); rng = np.random.default_rng(t)
                    idx = np.concatenate([rng.choice(np.where(ty == c)[0], SHOTS, replace=False)
                                          for c in sorted(set(ty.tolist())) if (ty == c).sum() >= SHOTS])
                    sx, sy = [tr_x[i] for i in idx], [lid[tr_y[i]] for i in idx]
                else:
                    sx, sy = tr_x, [lid[l] for l in tr_y]
                preds = train_eval(MODELS[m], sx, sy, te_x, len(labels), HP[a.setting], t, dev)
                acc = float(np.mean([p == g for p, g in zip(preds, gold)]))
                runs.append({"trial": t, "accuracy": acc, "seconds": time.time() - t0, "pred": preds})
                print(m, name, t, round(acc, 4), f"{time.time() - t0:.0f}s", flush=True)
            res.setdefault(m, {})[name] = {"labels": [str(l) for l in labels], "gold": gold,
                                           "test_idx": test_idx.tolist(), "runs": runs}
            json.dump(res, open(out_path, "w"))


if __name__ == "__main__":
    main()
