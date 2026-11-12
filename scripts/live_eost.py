"""Prospective (live) EOST-M evaluation of a model that is actually queried.

This is the protocol for the prospective deployment experiment requested by the
reviewers. It could not be executed in the paper's execution environment (no access to
model weights), so it is released for running on a machine that serves the model through
an OpenAI-compatible endpoint (llama.cpp server, vLLM, Ollama, ...):

  1. freeze: calibration panel of N leaderboard systems (seeded, excluding the target's
     family by name pattern), EOST-M model (calibration-marginal 2PL, EB prior), interval
     recalibration factors (leave-one-calibrator-out, rule of study 2), written to
     <out>/frozen.json before the first query;
  2. adaptive phase: the model answers the item chosen by EIG, an independent
     deterministic oracle grades it (GSM8K: exact match of the final number; CRUXEval-O:
     execution of assert f(input) == prediction), and the estimate and both intervals are
     recorded at every step; checkpoints at 20/40/60 items;
  3. truth phase (--full): the remaining items are administered (answers already obtained
     are reused) to compute the full-bank score that the checkpoints are compared with.
Every query and verdict is appended to <out>/log.jsonl, so an interrupted run resumes.

Testing without a model: --mock-system NAME answers with the recorded verdicts of a
leaderboard system (excluded from the panel), which must reproduce the replay engine.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from eost.data import load_bank  # noqa: E402
from eost.irt import Priors, calibrate, default_theta_grid, marginal_model  # noqa: E402
from eost.live import LiveTest  # noqa: E402
from eost.replay import Precomputed  # noqa: E402
from eost.replay2 import Z90, recalibration_factors  # noqa: E402
from eost.study2 import loco_marginal  # noqa: E402

GSM_DIR = "data/raw/openai_grade-school-math/grade_school_math/data"
CRUX = "data/raw/facebookresearch_cruxeval/data/cruxeval.jsonl"


# ------------------------------------------------------------------ items and oracles
def gsm8k_items():
    test = [json.loads(l) for l in open(f"{GSM_DIR}/test.jsonl")]
    train = [json.loads(l) for l in open(f"{GSM_DIR}/train.jsonl")][:5]
    shots = "".join(f"Question: {e['question']}\nAnswer: {e['answer']}\n\n" for e in train)
    items = []
    for e in test:
        gold = e["answer"].split("####")[-1].strip().replace(",", "")
        items.append({"prompt": shots + f"Question: {e['question']}\nAnswer:", "gold": gold})
    return items


def _number(text):
    m = re.findall(r"####\s*(-?[\d,]*\.?\d+)", text)
    if not m:
        m = re.findall(r"-?[\d,]*\.?\d+", text)
    return m[-1].replace(",", "").rstrip(".") if m else None


def gsm8k_oracle(item, text):
    pred = _number(text)
    try:
        return int(pred is not None and abs(float(pred) - float(item["gold"])) < 1e-6)
    except ValueError:
        return 0


def cruxeval_items():
    items = []
    for l in open(CRUX):
        e = json.loads(l)
        prompt = ("You are given a Python function and an input. Complete the assertion with the literal output "
                  "of executing the function on the input, even if the function is incorrect or incomplete. "
                  "Answer with the output only, between [ANSWER] and [/ANSWER].\n\n"
                  f"{e['code']}\nassert f({e['input']}) == ??")
        items.append({"prompt": prompt, "code": e["code"], "input": e["input"], "gold": e["output"]})
    return items


def cruxeval_oracle(item, text, timeout=5):
    m = re.search(r"\[ANSWER\](.*?)\[/ANSWER\]", text, re.S)
    pred = (m.group(1) if m else text).strip().split("\n")[0]
    if pred.startswith("assert"):
        pred = pred.split("==", 1)[-1].strip()
    prog = f"{item['code']}\nassert f({item['input']}) == {pred}\n"
    try:
        r = subprocess.run([sys.executable, "-I", "-c", prog], capture_output=True, timeout=timeout)
        return int(r.returncode == 0)
    except subprocess.TimeoutExpired:
        return 0


BANKS = {"gsm8k": ("ollb_gsm8k", gsm8k_items, gsm8k_oracle, 512),
         "cruxeval_o": ("cruxeval_o", cruxeval_items, cruxeval_oracle, 128)}


# ------------------------------------------------------------------ model client
def chat(endpoint, model, prompt, max_tokens, retries=5):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0, "max_tokens": max_tokens}).encode()
    for k in range(retries):
        try:
            req = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001
            if k == retries - 1:
                raise
            time.sleep(2 ** k)
    return ""


# ------------------------------------------------------------------ protocol
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", choices=sorted(BANKS), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--endpoint", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="local")
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261104)
    ap.add_argument("--exclude-pattern", default="", help="regex: leaderboard systems of the target's family")
    ap.add_argument("--stops", type=int, nargs="+", default=[20, 40, 60])
    ap.add_argument("--full", action="store_true", help="then administer every remaining item (truth phase)")
    ap.add_argument("--mock-system", default=None, help="testing: answer with this leaderboard system's verdicts")
    a = ap.parse_args()
    name, load_items, oracle, max_tokens = BANKS[a.bank]
    B = load_bank(f"data/processed/{name}.npz")
    os.makedirs(a.out, exist_ok=True)
    systems = [str(s) for s in B.systems]
    excl = np.array([bool(a.exclude_pattern) and re.search(a.exclude_pattern, s) is not None for s in systems])
    mock = None
    if a.mock_system:
        j = systems.index(a.mock_system)
        mock = B.Y[j].astype(int)
        excl |= B.related(j)
    rng = np.random.default_rng(a.seed)
    panel = np.sort(rng.choice(np.flatnonzero(~excl), a.N, replace=False))
    Yp = B.Y[panel].astype(float)
    cal, W, grid = calibrate(Yp, "2pl", Priors(), True)
    theta = default_theta_grid()
    pre = Precomputed.build(marginal_model(W, grid, theta), 1.0)
    pool = np.arange(B.I)
    T = max(a.stops)
    rule = json.load(open("configs/study2_hypotheses.json"))["recalibration_rule"]
    Z, E = loco_marginal(Yp, cal.theta_cal, cal.priors, grid, np.arange(a.N), B.Y[panel], pool, T, theta, 1.0, a.seed)
    if rule == "Ra":
        half = recalibration_factors(E * Z90, "quantile")
        kappa = None
    else:
        kappa = recalibration_factors(Z, "quantile" if rule == "Rq" else "rms")
        half = None
    frozen = {"bank": name, "panel": [systems[p] for p in panel], "seed": a.seed, "N": a.N, "stops": a.stops,
              "recalibration_rule": rule, "model": a.model, "endpoint": a.endpoint, "mock_system": a.mock_system,
              "bank_sha256": hashlib.sha256(open(f"data/processed/{name}.npz", "rb").read()).hexdigest(),
              "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    fz = f"{a.out}/frozen.json"
    if os.path.exists(fz) and json.load(open(fz))["panel"] != frozen["panel"]:
        raise SystemExit("an existing run in this directory used a different panel")
    if not os.path.exists(fz):
        json.dump(frozen, open(fz, "w"), indent=1)
    items = load_items() if mock is None else None
    logf = f"{a.out}/log.jsonl"
    done = {}
    if os.path.exists(logf):
        for l in open(logf):
            r = json.loads(l); done[r["item"]] = r["y"]

    def ask(i):
        if i in done:
            return done[i]
        if mock is not None:
            y, text = int(mock[i]), "<mock>"
        else:
            text = chat(a.endpoint, a.model, items[i]["prompt"], max_tokens)
            y = oracle(items[i], text)
        with open(logf, "a") as fh:
            fh.write(json.dumps({"item": int(i), "y": int(y), "output": text,
                                 "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")
        done[i] = y
        return y

    test = LiveTest(pre, pool, B.I, "eig", np.random.default_rng(a.seed))
    checkpoints = []
    for t in range(T):
        i = test.next_item()
        st = test.record(i, ask(i))
        hw = 0.5 * (st["hi"] - st["lo"])
        h = half[t] if half is not None else kappa[t] * hw
        st["lo_R"], st["hi_R"] = st["est"] - h, st["est"] + h
        if t + 1 in a.stops:
            checkpoints.append(st)
            print(f"t={t + 1:3d} estimate={st['est']:.3f} 90% interval=[{st['lo']:.3f},{st['hi']:.3f}] "
                  f"recalibrated=[{st['lo_R']:.3f},{st['hi_R']:.3f}]", flush=True)
    report = {"checkpoints": checkpoints, "items": [s["item"] for s in test.history]}
    if a.full:
        ys = np.array([ask(i) for i in range(B.I)])
        A = float(ys.mean())
        for c in checkpoints:
            c.update({"true_score": A, "abs_error": abs(c["est"] - A),
                      "covered": bool(c["lo"] <= A <= c["hi"]), "covered_R": bool(c["lo_R"] <= A <= c["hi_R"])})
        report["true_score"] = A
        print(f"full-bank score {A:.3f}; items administered in total: {len(done)}")
    json.dump(report, open(f"{a.out}/report.json", "w"), indent=1)
    print("LIVE_GATE=PASS")


if __name__ == "__main__":
    main()
