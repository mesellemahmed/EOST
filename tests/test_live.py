"""Tests of the live (prospective) protocol: engine equivalence, oracles, HTTP client."""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
from scipy.special import expit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from eost.irt import Priors, calibrate, default_theta_grid, marginal_model  # noqa: E402
from eost.live import LiveTest  # noqa: E402
from eost.replay import Precomputed, replay  # noqa: E402
import live_eost  # noqa: E402


def test_live_engine_reproduces_replay():
    rng = np.random.default_rng(0)
    th = rng.normal(size=30); a = np.exp(rng.normal(0, .4, 150)); b = rng.normal(0, 1, 150)
    Y = (rng.random((30, 150)) < expit(a * (th[:, None] - b))).astype(int)
    cal, W, grid = calibrate(Y[:8].astype(float), "2pl", Priors(), True)
    pre = Precomputed.build(marginal_model(W, grid, default_theta_grid()), 1.0)
    y = Y[20]
    pool = np.arange(150)
    out = replay(y, pre, pool, np.array([], dtype=int), "eig", 40, np.random.default_rng(5))
    lt = LiveTest(pre, pool, 150, "eig", np.random.default_rng(5))
    for t in range(40):
        i = lt.next_item()
        st = lt.record(i, int(y[i]))
        assert i == out["items"][t]
        assert np.isclose(st["est"], out["est"][t]) and np.isclose(st["lo"], out["lo"][t])


def test_gsm8k_oracle_extracts_final_number():
    item = {"gold": "1234"}
    assert live_eost.gsm8k_oracle(item, "so the total is 1,234.\n#### 1,234")
    assert live_eost.gsm8k_oracle(item, "The answer is 1234.")
    assert not live_eost.gsm8k_oracle(item, "#### 1235")


def test_cruxeval_oracle_executes_the_assertion():
    item = {"code": "def f(x):\n    return x * 2", "input": "3"}
    assert live_eost.cruxeval_oracle(item, "[ANSWER]6[/ANSWER]") == 1
    assert live_eost.cruxeval_oracle(item, "[ANSWER]7[/ANSWER]") == 0
    assert live_eost.cruxeval_oracle(item, "[ANSWER]assert f(3) == 6[/ANSWER]") == 1


def test_chat_client_against_a_local_openai_compatible_server():
    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers["Content-Length"])
            req = json.loads(self.rfile.read(n))
            body = json.dumps({"choices": [{"message": {"content": "echo:" + req["messages"][0]["content"]}}]})
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(body.encode())

        def log_message(self, *a):
            pass
    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        out = live_eost.chat(f"http://127.0.0.1:{srv.server_port}/v1", "m", "hello", 8)
        assert out == "echo:hello"
    finally:
        srv.shutdown()
