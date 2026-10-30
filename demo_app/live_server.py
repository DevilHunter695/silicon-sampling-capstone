"""Local live-inference server for the demo page (Apple Silicon, MLX).

Loads Qwen2.5-7B-Instruct (4-bit MLX) plus our QLoRA adapter and returns the real next-digit probability
distribution for any respondent-question pair. The demo page calls it on http://127.0.0.1:8765.

Setup (once, needs internet):   pip3 install mlx-lm safetensors numpy
Run before the demo:            python3 demo_app/live_server.py --adapter /path/to/adapter_dir
The first start downloads the 4-bit base model (~4.3 GB) and converts the adapter. Later starts are offline.

Honest caveat: training used a bitsandbytes NF4 base in fp16/bf16; this runs an MLX 4-bit base. The adapter weights
are identical but the base quantisation differs, so live probabilities can differ slightly from the stored full-fold run.
"""
import argparse, json, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.prompts.templates import build_prompt            # same P2 builder used for training and scoring
from src.inference.prompting import build_answer_instruction

BASE = "mlx-community/Qwen2.5-7B-Instruct-4bit"
STATE = {"model": None, "tok": None, "ready": False, "error": None, "adapter": None}


def make_prompt(profile: dict, question: str, options: list) -> str:
    opts = "\n".join(f"{i + 1}. {o}" for i, o in enumerate(options))
    labels = [str(i + 1) for i in range(len(options))]
    return build_prompt("P2", question, opts, **profile) + build_answer_instruction(labels) + "\n\n"


def convert_adapter(peft_dir: Path, out_dir: Path):
    """PEFT LoRA (A: r x in, B: out x r) -> mlx_lm LoRA (lora_a: in x r, lora_b: r x out)."""
    import mlx.core as mx
    cfg = json.load(open(peft_dir / "adapter_config.json"))
    w = mx.load(str(peft_dir / "adapter_model.safetensors"))          # mx handles the bfloat16 tensors
    out = {}
    for k, v in w.items():
        k2 = k.replace("base_model.model.", "", 1)
        if ".lora_A." in k2: out[k2.replace(".lora_A.weight", ".lora_a")] = v.T.astype(mx.float16)
        elif ".lora_B." in k2: out[k2.replace(".lora_B.weight", ".lora_b")] = v.T.astype(mx.float16)
    layers = sorted({int(k.split(".layers.")[1].split(".")[0]) for k in out})
    out_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out_dir / "adapters.safetensors"), out)
    json.dump({"fine_tune_type": "lora", "num_layers": max(layers) + 1,
               "lora_parameters": {"rank": cfg["r"], "scale": cfg["lora_alpha"] / cfg["r"], "dropout": 0.0,
                                   "keys": [f"self_attn.{m}" for m in cfg["target_modules"]]}},
              open(out_dir / "adapter_config.json", "w"))
    return len(out)


def load_model(adapter_dir: Path):
    try:
        from mlx_lm import load
        conv = adapter_dir.parent / (adapter_dir.name + "_mlx")
        if not (conv / "adapters.safetensors").exists():
            print("converting adapter ->", conv, convert_adapter(adapter_dir, conv), "tensors", flush=True)
        STATE["model"], STATE["tok"] = load(BASE, adapter_path=str(conv))
        score(make_prompt({"sex": "Female", "age": "35-44"}, "Warm-up question?", ["A", "B"]), 2)   # compile kernels
        STATE.update(ready=True, adapter=str(adapter_dir))
        print("model ready", flush=True)
    except Exception as e:                                   # surfaced in /health so the page can say why
        STATE["error"] = repr(e); print("LOAD FAILED:", e, flush=True)


def score(prompt: str, n: int):
    import mlx.core as mx
    tok, model = STATE["tok"], STATE["model"]
    ids = tok.encode(prompt)
    logits = model(mx.array([ids]))[0, -1].astype(mx.float32)
    digit_ids = [tok.encode(str(i + 1))[0] for i in range(n)]
    p = mx.softmax(logits[mx.array(digit_ids)])
    return [float(x) for x in p.tolist()]


class H(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        for k, v in (("Content-Type", "application/json"), ("Access-Control-Allow-Origin", "*"),
                     ("Access-Control-Allow-Headers", "Content-Type"), ("Access-Control-Allow-Private-Network", "true"),
                     ("Content-Length", str(len(b)))): self.send_header(k, v)
        self.end_headers(); self.wfile.write(b)

    def do_OPTIONS(self): self._send(204, {})

    def do_GET(self):
        self._send(200, {"ready": STATE["ready"], "error": STATE["error"], "base": BASE, "adapter": STATE["adapter"]})

    def do_POST(self):
        if not STATE["ready"]: return self._send(503, {"error": STATE["error"] or "model still loading"})
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        t = time.time()
        prompt = make_prompt(req["profile"], req["question"], req["options"])
        p = score(prompt, len(req["options"]))
        self._send(200, {"probs": p, "ms": round((time.time() - t) * 1000), "prompt": prompt})

    def log_message(self, *a): pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--adapter", required=True); ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    import threading
    threading.Thread(target=load_model, args=(Path(a.adapter),), daemon=True).start()
    print(f"live server on http://127.0.0.1:{a.port}  (model loading in background)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
