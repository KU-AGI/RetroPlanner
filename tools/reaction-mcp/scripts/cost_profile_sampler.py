#!/usr/bin/env python
"""Sample every compute counter this box exposes, once a minute, to one JSONL.

WHY A SAMPLER AND NOT A POST-HOC READ. Every counter here is a MONOTONIC TOTAL on a
long-lived server: vLLM replicas and the SSR fleet outlive a single arm, so reading them
once at the end gives the sum over everything that ever ran against them. The per-arm
answer -- which is the one the compute question needs -- exists only as a DIFFERENCE
between two samples, and only if someone was sampling while the arm ran. There is no way
to recover it afterwards.

WHAT EACH COUNTER IS, and the two that are easy to misread:

  vllm:prompt_tokens_total        ALL prompt tokens, including the ones a prefix-cache hit
                                  served for free. This is NOT prefill compute.
  vllm:prompt_tokens_cached_total the free half. prefill_compute = total - cached, and at a
                                  43% hit rate that is a 1.75x error if you use the total.
  vllm:generation_tokens_total    decode tokens; each costs a full forward pass.
  ssr 9021 hit/miss               the draw cache in front of the single-step fleet. `miss`
                                  is the only part that costs a root_aligned forward pass;
                                  `hit` is a dict lookup. A run whose menus are 90% cached
                                  has spent almost no single-step compute, and counting
                                  menu REQUESTS instead of misses overstates it ~10x.

Energy is sampled, not derived: GPU power is integrated over the sample interval, which is
the one number that needs no model-architecture assumption at all.
"""
import json
import os

# The two SSR ports are a machine layout, not a property of the measurement: 9021 is the
# menu cache in front of 9020's pool proxy (config/env.sh names them). Read them from the
# environment so a different box needs no edit here.
_PORT_MENU = os.environ.get("RP_PORT_LOCALRETRO", "9021")
_PORT_SSR = os.environ.get("RP_PORT_RSMILES", "9020"), os, re, sys, time, urllib.request

OUT = sys.argv[1] if len(sys.argv) > 1 else "logs/cost_profile/profile.jsonl"
CHAIN = os.environ.get("CHAIN_LOG", "")
# A runner that is not the rp chain names its own arm in a file it rewrites between arms.
# Reading a label beats parsing someone else's log, and it is what makes a sample
# attributable at all -- an unlabelled sample cannot be differenced into a per-arm cost.
ARM_FILE = os.environ.get("ARM_FILE", "")
LLM_PORTS = [int(x) for x in os.environ.get("LLM_PORTS", "8300 8301 8302 8303 8304 8305 8306 8307").split()]
PERIOD = float(os.environ.get("PERIOD", "60"))
ROWS_GLOB = os.environ.get("ROWS_GLOB", "logs/rpfix_shards/out*.jsonl")

_num = r"([0-9.eE+-]+)"


def get(url, timeout=5):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read().decode()
    except Exception:
        return None


def vllm(port):
    t = get(f"http://127.0.0.1:{port}/metrics")
    if not t:
        return None
    def m(name):
        g = re.search(rf"^vllm:{name}\{{[^}}]*\}} {_num}$", t, re.M)
        return float(g.group(1)) if g else None
    # request_success_total is split by finish reason, so every label line must be summed.
    succ = sum(float(x) for x in re.findall(rf"^vllm:request_success_total\{{[^}}]*\}} {_num}$", t, re.M))
    # TIME IS NOT FLOPs AND THE TWO DISAGREE BY AN ORDER OF MAGNITUDE HERE. Prefill carries
    # almost all of the arithmetic and decode carries almost all of the seconds: measured on
    # these replicas, prefill 12.6e3 s against decode 293.1e3 s, because decode is one token
    # at a time and memory-bound while prefill is a dense matmul over the whole prompt. A
    # cost table that reports only one of them answers only half the question, so both sums
    # are carried, plus the queue time that is neither.
    def hsum(name):
        g = re.search(rf"^vllm:{name}_sum\{{[^}}]*\}} {_num}$", t, re.M)
        return float(g.group(1)) if g else None
    return {"port": port, "req": succ, "prompt_tok": m("prompt_tokens_total"),
            "prompt_cached_tok": m("prompt_tokens_cached_total"),
            "gen_tok": m("generation_tokens_total"),
            "running": m("num_requests_running"), "waiting": m("num_requests_waiting"),
            "e2e_s": hsum("e2e_request_latency_seconds"),
            "prefill_s": hsum("request_prefill_time_seconds"),
            "decode_s": hsum("request_decode_time_seconds"),
            "infer_s": hsum("request_inference_time_seconds"),
            "queue_s": hsum("request_queue_time_seconds"),
            "ttft_s": hsum("time_to_first_token_seconds"),
            # the engine's OWN count of prompt tokens it actually computed KV for; this is
            # the prefill work, and it needs no total-minus-cached subtraction
            "prefill_kv_tok": hsum("request_prefill_kv_computed_tokens")}


def jget(url):
    t = get(url)
    if not t:
        return None
    try:
        return json.loads(t)
    except Exception:
        return None


def gpus():
    import subprocess
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,power.draw,memory.used",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return []
    out = []
    for line in o.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) == 4:
            out.append({"i": int(p[0]), "util": float(p[1]), "w": float(p[2]), "mem": float(p[3])})
    return out


def arm():
    if ARM_FILE:
        try:
            return open(ARM_FILE).read().strip() or None
        except Exception:
            return None
    if not CHAIN or not os.path.exists(CHAIN):
        return None
    cur = None
    for l in open(CHAIN, errors="ignore"):
        g = re.search(r"===== (\S+) ->", l)
        if g:
            cur = g.group(1)
        if "chain complete" in l:
            cur = "done"
    return cur


def rows():
    import glob
    n = 0
    for f in glob.glob(ROWS_GLOB):
        try:
            with open(f, errors="ignore") as fh:
                n += sum(1 for x in fh if x.strip())
        except Exception:
            pass
    return n


os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
print(f"# cost profiler -> {OUT}  every {PERIOD:.0f}s  ports={LLM_PORTS}", flush=True)
while True:
    rec = {"ts": time.time(), "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "arm": arm(), "rows": rows(),
           "vllm": [v for v in (vllm(p) for p in LLM_PORTS) if v],
           "gpu": gpus()}
    h = jget(f"http://127.0.0.1:{_PORT_MENU}/health")
    if h:
        rec["ssr_cache"] = {k: h.get(k) for k in ("hit", "miss", "upstream_fail", "cache_entries")}
    # Forward replicas only grew counters on 2026-11-08; a fleet started before that
    # answers /health without "stats" and this stays absent rather than reading 0, which
    # would otherwise look like a run that did no round-trip scoring at all.
    fwd = {"requests": 0, "reactions": 0, "in_tok": 0, "out_tok": 0}
    seen = False
    for fp in range(8090, 8098):
        hh = jget(f"http://127.0.0.1:{fp}/health")
        st = (hh or {}).get("stats")
        if st:
            seen = True
            for k in fwd:
                fwd[k] += st.get(k, 0)
    if seen:
        rec["forward"] = fwd

    h = jget(f"http://127.0.0.1:{_PORT_SSR}/health")
    if h:
        reps = h.get("replicas") or []
        # inflight is a GAUGE, so integrating it over the sample interval gives SSR
        # request-seconds -- the single-step fleet has no latency counter of its own and
        # this is the only way to put its time on the same axis as the LLM's.
        rec["ssr_pool"] = {"n_live": h.get("n_live"),
                           "ok": sum(r.get("ok", 0) for r in reps),
                           "fail": sum(r.get("fail", 0) for r in reps),
                           "inflight": sum(r.get("inflight", 0) for r in reps)}
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    time.sleep(PERIOD)
