"""Build binary item-response matrices from pinned public sources.

Runs inside the container with no network. Output: one ``.npz`` per benchmark
in ``data/processed`` plus ``data/processed/MANIFEST.json`` holding SHA-256
hashes of every raw input and processed output.

Matrix convention: ``Y[j, i]`` in {0, 1} is the deterministic oracle verdict of
system ``j`` on item ``i`` (1 = correct / resolved). Systems carry a ``family``
label used to exclude related systems from calibration panels.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import zipfile


import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from eost.safeio import safe_load_pickle, sha256_file  # noqa: E402

RAW = "data/raw"
OUT = "data/processed"
SUBSAMPLE_SEED = 20261029  # fixed before any analysis; documented in the paper
MAX_ITEMS_LARGE = 2000     # MMLU and HellaSwag are randomly subsampled to this size


def save(name, Y, systems, families, items, meta, raw_files, manifest):
    Y = np.asarray(Y, dtype=np.uint8)
    assert Y.shape == (len(systems), len(items)), (name, Y.shape)
    assert set(np.unique(Y)) <= {0, 1}, name
    assert len(set(systems)) == len(systems), f"duplicate systems in {name}"
    path = os.path.join(OUT, f"{name}.npz")
    np.savez_compressed(path, Y=Y, systems=np.array(systems), families=np.array(families),
                        items=np.array(items), meta=json.dumps(meta))
    acc = Y.mean(1)
    manifest["datasets"][name] = {
        "file": path, "sha256": sha256_file(path), "n_systems": int(Y.shape[0]),
        "n_items": int(Y.shape[1]), "n_families": int(len(set(families))),
        "mean_accuracy": float(Y.mean()), "accuracy_range": [float(acc.min()), float(acc.max())],
        "items_never_correct": int((Y.sum(0) == 0).sum()),
        "items_always_correct": int((Y.sum(0) == Y.shape[0]).sum()),
        "raw_inputs": {f: sha256_file(f) for f in sorted(set(raw_files))}, "meta": meta,
    }
    print(f"[{name}] systems={Y.shape[0]} items={Y.shape[1]} families={len(set(families))} "
          f"mean_acc={Y.mean():.3f} range=[{acc.min():.3f},{acc.max():.3f}]")


# ---------------------------------------------------------------- Open LLM Leaderboard
def ollb_family(model_path: str) -> str:
    name = model_path.split("details_", 1)[1]
    return name.split("__", 1)[0].lower()  # organisation namespace


def build_ollb(manifest):
    f = f"{RAW}/felipemaiapolo_tinyBenchmarks/tutorials/data/lb.pickle"
    d = safe_load_pickle(f)
    models = [m.split("details_", 1)[1].replace("__", "/") for m in d["models"]]
    fams = [ollb_family(m) for m in d["models"]]
    rng = np.random.default_rng(SUBSAMPLE_SEED)
    spec = {
        "ollb_gsm8k": ["harness_gsm8k_5"],
        "ollb_arc": ["harness_arc_challenge_25"],
        "ollb_winogrande": ["harness_winogrande_5"],
        "ollb_hellaswag": ["harness_hellaswag_10"],
        "ollb_truthfulqa": ["harness_truthfulqa_mc_0"],
        "ollb_mmlu": sorted(k for k in d["data"] if "hendrycksTest" in k),
    }
    for name, keys in spec.items():
        blocks, items = [], []
        for k in keys:
            c = np.asarray(d["data"][k]["correctness"], dtype=float)  # (items, models)
            blocks.append(c)
            items += [f"{k}:{i}" for i in range(c.shape[0])]
        C = np.concatenate(blocks, axis=0).T  # (models, items)
        rule = "binary correctness from Open LLM Leaderboard v1 details"
        if name == "ollb_truthfulqa":
            C = (C >= 0.5).astype(float)
            rule = "MC2 normalized probability on true answers >= 0.5"
        idx = np.arange(C.shape[1])
        if C.shape[1] > MAX_ITEMS_LARGE:
            idx = np.sort(rng.choice(C.shape[1], size=MAX_ITEMS_LARGE, replace=False))
            rule += f"; uniform random subset of {MAX_ITEMS_LARGE} items (seed {SUBSAMPLE_SEED})"
        meta = {"source": "tinybenchmarks", "binarization": rule,
                "family_rule": "organisation namespace of the Hugging Face model id",
                "domain": {"ollb_gsm8k": "math word problems (exact match)"}.get(name, "multiple choice (log-likelihood)")}
        save(name, C[:, idx], models, fams, [items[i] for i in idx], meta, [f], manifest)


# ---------------------------------------------------------------- SWE-bench
def norm_model(m: str) -> str:
    m = m.lower().strip()
    m = re.sub(r"[-_ ]?20\d{2}[-_]?\d{2}[-_]?\d{2}$", "", m)  # drop date suffix
    return m


def build_swe(manifest, split):
    base = f"{RAW}/swe_exp/evaluation/{split}"
    raw_files, rows = [], []
    for sub in sorted(os.listdir(base)):
        meta_f = os.path.join(base, sub, "metadata.yaml")
        res_f = os.path.join(base, sub, "results", "results.json")
        det_f = os.path.join(base, sub, "per_instance_details.json")
        resolved, universe = None, set()
        if os.path.exists(res_f):
            r = json.load(open(res_f))
            resolved = set(r.get("resolved", []))
            for v in r.values():
                if isinstance(v, list):
                    universe |= set(v)
            raw_files.append(res_f)
        elif os.path.exists(det_f):
            r = json.load(open(det_f))
            resolved = {k for k, v in r.items() if v.get("resolved")}
            universe = set(r)
            raw_files.append(det_f)
        if resolved is None:
            continue
        models = []
        if os.path.exists(meta_f):
            raw_files.append(meta_f)
            md = yaml.safe_load(open(meta_f)) or {}
            models = (md.get("tags") or {}).get("model") or []
            if isinstance(models, str):
                models = [models]
        models = sorted({norm_model(m) for m in models}) or [f"unknown:{sub}"]
        rows.append((sub, resolved, universe, models))
    # canonical item set: union of instance ids listed by any submission (many
    # results.json files only list a few outcome categories); instances that a
    # submission does not list as resolved count as unresolved (official scoring).
    union = set()
    for _, _, u, _ in rows:
        union |= u
    expected = {"verified": 500, "lite": 300}[split]
    items = sorted(union)
    if len(items) != expected:
        raise SystemExit(f"SWE {split}: expected {expected} items, found {len(items)}")
    systems, fams, Y = [], [], []
    for sub, resolved, _, models in rows:
        if not resolved <= set(items):
            resolved = resolved & set(items)
        systems.append(sub)
        fams.append("+".join(models))
        Y.append([1 if x in resolved else 0 for x in items])
    meta = {"source": "swe_bench_experiments", "binarization": "instance resolved by the official harness (unresolved, errors and missing patches count as 0)",
            "family_rule": "set of underlying LLMs declared in metadata.yaml tags.model; systems sharing any LLM are excluded from the target's panel",
            "family_models": {s: f.split("+") for s, f in zip(systems, fams)},
            "domain": "agentic software engineering (execution-based tests)"}
    save(f"swe_{split}", np.array(Y), systems, fams, items, meta, raw_files, manifest)


# ---------------------------------------------------------------- LiveCodeBench
LCB_FAMILY = [
    (r"^o3-mini", "openai-o3-mini"), (r"^o4-mini", "openai-o4-mini"), (r"^o3 ", "openai-o3"),
    (r"^gpt-4o", "openai-gpt-4o"), (r"^gpt-4-turbo", "openai-gpt-4-turbo"),
    (r"^claude-(sonnet|opus)-4", "anthropic-claude-4"), (r"^claude-3", "anthropic-claude-3"),
    (r"^gemini-2.5-pro", "google-gemini-2.5-pro"), (r"^gemini-2.5-flash", "google-gemini-2.5-flash"),
    (r"^deepseek", "deepseek"), (r"^qwen", "qwen"), (r"^grok", "xai-grok"),
    (r"nemotron", "nvidia-nemotron"), (r"^xbai", "xbai"), (r"^exaone", "lg-exaone"),
]


def lcb_family(m: str) -> str:
    s = m.lower()
    for pat, fam in LCB_FAMILY:
        if re.search(pat, s):
            return fam
    return s


def build_lcb(manifest):
    f = f"{RAW}/LiveCodeBench_livecodebench.github.io/build/performances_generation.json"
    d = json.load(open(f))
    perf = d["performances"]
    models = sorted({p["model"] for p in perf})
    items = sorted({p["question_id"] for p in perf})
    mi, ii = {m: k for k, m in enumerate(models)}, {q: k for k, q in enumerate(items)}
    Y = np.full((len(models), len(items)), 255, dtype=np.int16)
    for p in perf:
        Y[mi[p["model"]], ii[p["question_id"]]] = 1 if p["pass@1"] >= 50 else 0
    if (Y == 255).any():
        raise SystemExit("LiveCodeBench matrix has missing cells")
    diff = {p["question_id"]: p["difficulty"] for p in perf}
    meta = {"source": "livecodebench_site", "binarization": "pass@1 (fraction of sampled programs passing all tests) >= 50%",
            "family_rule": "manual mapping of reasoning-effort / version variants of the same model line",
            "item_difficulty_label": [diff[q] for q in items], "domain": "code generation (execution-based tests)"}
    save("livecodebench", Y, models, [lcb_family(m) for m in models], items, meta, [f], manifest)


# ---------------------------------------------------------------- CRUXEval
def crux_family(m: str) -> str:
    s = m.lower()
    for pre in ["codellama", "deepseek", "wizard", "phind", "gpt-3.5", "gpt-4", "starcoder",
                "mistral", "mixtral", "magicoder", "codetulu", "semcoder", "phi"]:
        if s.startswith(pre):
            return pre
    return s.split("-")[0]


def build_crux(manifest):
    f = f"{RAW}/facebookresearch_cruxeval/samples/evaluation_results.zip"
    z = zipfile.ZipFile(f)
    for task in ["output", "input"]:
        rows = {}
        for n in z.namelist():
            m = re.match(rf"evaluation_results/(.+)_temp0\.2_{task}\.json$", n)
            if not m:
                continue
            d = json.loads(z.read(n))
            sc = d["raw_scored_generations"]
            rows[m.group(1)] = {k: float(np.mean(v)) for k, v in sc.items()}
        models = sorted(rows)
        items = sorted(set.intersection(*(set(r) for r in rows.values())), key=lambda s: int(s.split("_")[1]))
        if len(items) != 800:
            raise SystemExit(f"CRUXEval-{task}: expected 800 items, got {len(items)}")
        Y = np.array([[1 if rows[m][q] >= 0.5 else 0 for q in items] for m in models])
        meta = {"source": "cruxeval", "binarization": "fraction of temperature-0.2 samples judged correct by execution >= 0.5",
                "family_rule": "model line prefix (e.g. all CodeLlama variants form one family)",
                "domain": f"code reasoning ({task} prediction, execution-checked)"}
        save(f"cruxeval_{task[0]}", Y, models, [crux_family(m) for m in models], items, meta, [f], manifest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    manifest = {"subsample_seed": SUBSAMPLE_SEED, "max_items_large": MAX_ITEMS_LARGE, "datasets": {}}
    builders = {"ollb": build_ollb, "swe": lambda m: (build_swe(m, "verified"), build_swe(m, "lite")),
                "lcb": build_lcb, "crux": build_crux}
    for k, fn in builders.items():
        if args.only and k not in args.only:
            continue
        fn(manifest)
    with open(os.path.join(OUT, "MANIFEST.json"), "w") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)
    print("DATA_GATE=PASS", len(manifest["datasets"]), "datasets")


if __name__ == "__main__":
    main()
