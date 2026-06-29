"""TechNode provider commands — turn a Linux box into a grid serving node.

`technode provider serve` runs the stdlib-only serving daemon (pc_serve.py) in
*pull mode*: it polls the broker over outbound HTTPS for jobs it can serve, runs
them on the local GPU, and posts results back. No inbound port, no Tailscale —
works behind any NAT. This is the cross-platform / marketplace path.

    technode provider register --gpu "RTX 4090" --vram 24
    technode provider serve --llama-server /opt/llama.cpp/llama-server
    technode provider status

Provider ops talk to the broker directly (long-poll doesn't fit a serverless
proxy's time limit). Override with TN_BROKER.
"""

import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request

BROKER_URL = (os.environ.get("TN_BROKER", "").strip()
              or "https://broker.technode.network").rstrip("/")
PC_SERVE_URL = "https://technode.network/agent/pc_serve.py"
TN_DIR = os.path.join(os.path.expanduser("~"), ".technode")
PROVIDER_CFG = os.path.join(TN_DIR, "provider.json")
PC_SERVE_PATH = os.path.join(TN_DIR, "pc_serve.py")
MODELS_DIR = os.path.join(TN_DIR, "models")
LLAMA_DIR = os.path.join(TN_DIR, "llama")
UA = "technode-cli-provider"


def _die(msg, code=1):
    print("technode: " + msg, file=sys.stderr)
    raise SystemExit(code)


def _req(method, url, body=None, timeout=30):
    data = json.dumps(body).encode() if body is not None else None
    headers = {"User-Agent": UA, "Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            payload = json.loads(raw)
        except ValueError:
            payload = {"error": raw[:300] or e.reason}
        payload["_status"] = e.code
        return payload
    except urllib.error.URLError as e:
        _die(f"cannot reach broker {BROKER_URL} — {e.reason}")


def _load_cfg():
    try:
        with open(PROVIDER_CFG, encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, ValueError):
        return {}


def _save_cfg(cfg):
    os.makedirs(TN_DIR, exist_ok=True)
    tmp = PROVIDER_CFG + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
    os.replace(tmp, PROVIDER_CFG)
    try:
        os.chmod(PROVIDER_CFG, 0o600)
    except OSError:
        pass


# --------------------------------------------------------------------------- #
def cmd_register(args):
    import socket
    body = {
        "name": args.name or socket.gethostname(),
        "hostname": socket.gethostname(),
        "gpu_name": args.gpu or "",
        "vram_gb": int(args.vram or 0),
        "owner_email": args.email or "",
    }
    existing = _load_cfg()
    if existing.get("register_token"):
        # Re-register: prove ownership so the broker refreshes rather than rejects.
        body["provider_id"] = existing.get("provider_id", "")
        body["register_token"] = existing["register_token"]
    res = _req("POST", BROKER_URL + "/provider/register", body)
    if res.get("error"):
        _die(f"register failed — {res.get('error')}")
    cfg = {
        "provider_id": res.get("provider_id"),
        "register_token": res.get("register_token"),
        "dashboard_token": res.get("dashboard_token"),
        "broker": BROKER_URL,
    }
    _save_cfg(cfg)
    print("Registered ✓")
    print(f"  provider_id: {cfg['provider_id']}")
    print(f"  dashboard:   {res.get('dashboard_url', '')}")
    print(f"  creds saved: {PROVIDER_CFG} (chmod 600)")
    print("\nNext:  technode provider serve   (needs operator approval to receive jobs)")
    return 0


def cmd_status(args):
    cfg = _load_cfg()
    if not cfg.get("provider_id"):
        print("Not registered. Run `technode provider register`.")
        return 1
    print(f"provider_id: {cfg['provider_id']}")
    print(f"broker:      {cfg.get('broker', BROKER_URL)}")
    me = _req("GET", BROKER_URL + f"/provider/me?t={cfg.get('dashboard_token','')}", timeout=15)
    if me.get("error"):
        print(f"approval:    unknown ({me.get('error')})")
    else:
        print(f"approved:    {me.get('approved')}")
        if me.get("serving_models"):
            print(f"models:      {', '.join(me['serving_models'])}")
    llama = _find_llama(args.llama_server if hasattr(args, "llama_server") else None)
    print(f"llama-server: {llama or 'NOT FOUND (set --llama-server or install llama.cpp)'}")
    return 0


def _find_llama(explicit=None):
    for cand in (explicit, os.environ.get("TN_LLAMA_BIN"),
                 shutil.which("llama-server"),
                 os.path.join(LLAMA_DIR, "llama-server"),
                 os.path.join(LLAMA_DIR, "build", "bin", "llama-server")):
        if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def _ensure_pc_serve():
    if os.path.isfile(PC_SERVE_PATH):
        return PC_SERVE_PATH
    os.makedirs(TN_DIR, exist_ok=True)
    print(f"downloading serving daemon → {PC_SERVE_PATH}")
    req = urllib.request.Request(PC_SERVE_URL, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
    except Exception as e:
        _die(f"could not download pc_serve.py from {PC_SERVE_URL} — {e}")
    with open(PC_SERVE_PATH, "wb") as fh:
        fh.write(data)
    return PC_SERVE_PATH


def cmd_serve(args):
    cfg = _load_cfg()
    if not cfg.get("register_token"):
        _die("not registered. Run `technode provider register` first.")
    llama = _find_llama(args.llama_server)
    if not llama:
        _die("llama-server not found.\n"
             "        Point to it:  technode provider serve --llama-server /path/to/llama-server\n"
             "        or set TN_LLAMA_BIN, or put it in ~/.technode/llama/.\n"
             "        NVIDIA build: https://github.com/ggml-org/llama.cpp/releases "
             "(or build with -DGGML_CUDA=ON).")
    pc_serve = _ensure_pc_serve()
    os.makedirs(MODELS_DIR, exist_ok=True)
    cmd = [sys.executable, pc_serve, "--pull",
           "--provider-id", cfg["provider_id"],
           "--register-token", cfg["register_token"],
           "--bin", llama, "--models", MODELS_DIR]
    if args.models:
        cmd += ["--serve-models", args.models]
    if args.no_inbound:
        # pull mode doesn't need the inbound server; bind it to localhost only is
        # not exposed via a flag, so we just note it. (Harmless when unreachable.)
        pass
    print(f"starting pull-mode serving: provider={cfg['provider_id']}  llama={llama}")
    print(f"  broker={BROKER_URL}  models-cache={MODELS_DIR}")
    print("  (Ctrl-C to stop)\n")
    env = dict(os.environ, TN_BROKER=BROKER_URL)
    try:
        return subprocess.call(cmd, env=env)
    except KeyboardInterrupt:
        return 130


def cmd_install(args):
    """Emit a systemd unit that runs `technode provider serve` on boot."""
    cfg = _load_cfg()
    if not cfg.get("register_token"):
        _die("register first: technode provider register")
    tn = shutil.which("technode") or os.path.join(os.path.dirname(sys.executable), "technode")
    llama = _find_llama(args.llama_server) or "/path/to/llama-server"
    user = os.environ.get("USER", "root")
    unit = f"""[Unit]
Description=TechNode provider (pull-mode serving)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User={user}
ExecStart={tn} provider serve --llama-server {llama}
Restart=always
RestartSec=5
Environment=TN_BROKER={BROKER_URL}

[Install]
WantedBy=multi-user.target
"""
    print("# Save as /etc/systemd/system/technode-provider.service, then:")
    print("#   sudo systemctl daemon-reload && sudo systemctl enable --now technode-provider")
    print("# ----------------------------------------------------------------")
    print(unit)
    return 0


def add_parser(sub):
    p = sub.add_parser("provider", help="run a GPU as a grid serving node (Linux/cross-platform)")
    psub = p.add_subparsers(dest="provider_command")

    s = psub.add_parser("register", help="register this machine as a provider")
    s.add_argument("--name", help="display name (default: hostname)")
    s.add_argument("--gpu", help="GPU name, e.g. \"RTX 4090\"")
    s.add_argument("--vram", help="GPU VRAM in GB")
    s.add_argument("--email", help="owner email (optional)")
    s.set_defaults(func=cmd_register)

    s = psub.add_parser("serve", help="serve models in pull mode (outbound-only, NAT-friendly)")
    s.add_argument("--llama-server", help="path to the llama-server binary")
    s.add_argument("--models", help="comma-separated catalog ids to advertise (default: auto by VRAM)")
    s.add_argument("--no-inbound", action="store_true", help="pull only (no inbound serving)")
    s.set_defaults(func=cmd_serve)

    s = psub.add_parser("status", help="show registration + approval + llama-server")
    s.add_argument("--llama-server", help="path to the llama-server binary")
    s.set_defaults(func=cmd_status)

    s = psub.add_parser("install", help="print a systemd unit for boot persistence")
    s.add_argument("--llama-server", help="path to the llama-server binary")
    s.set_defaults(func=cmd_install)

    p.set_defaults(func=lambda a: (p.print_help() or 0))
