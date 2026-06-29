"""TechNode CLI.

A tiny, dependency-free client for TechNode's compressed open-model GPU grid.

    pip install technode
    technode login              # paste your tn_test_... key (or set TECHNODE_API_KEY)
    technode models            # list the compressed catalog
    technode infer "Explain quantization in one line." --model qwen2.5-7b

Design notes
------------
* Pure stdlib (urllib) — no third-party deps, so it runs anywhere Python does.
* The authenticated path goes through the website API (https://technode.network),
  which validates the key + charges credit server-side and forwards to the GPU
  broker. Secrets never touch the CLI beyond the user's own key.
* The model catalog is public, so `models` reads it without auth.
"""

import argparse
import getpass
import json
import os
import sys
import time
import urllib.error
import urllib.request

from . import __version__

DEFAULT_BASE_URL = "https://technode.network"
# Public, unauthenticated catalog (read-only). Used as a fallback if the website
# proxy (/api/models) is unavailable.
CATALOG_FALLBACK_URL = "https://broker.technode.network/models/available"
CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".technode")
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.json")
USER_AGENT = "technode-cli/" + __version__


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def _load_config() -> dict:
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            return data if isinstance(data, dict) else {}
    except (FileNotFoundError, ValueError, OSError):
        return {}


def _save_config(cfg: dict) -> None:
    os.makedirs(CONFIG_DIR, exist_ok=True)
    tmp = CONFIG_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
    os.replace(tmp, CONFIG_PATH)
    try:
        os.chmod(CONFIG_PATH, 0o600)  # the key is a secret; lock the file down
    except OSError:
        pass


def _api_key() -> str:
    """Resolve the API key: env wins, then the saved config."""
    return (os.environ.get("TECHNODE_API_KEY", "").strip()
            or str(_load_config().get("api_key", "")).strip())


def _base_url() -> str:
    return (os.environ.get("TECHNODE_BASE_URL", "").strip()
            or str(_load_config().get("base_url", "")).strip()
            or DEFAULT_BASE_URL).rstrip("/")


# --------------------------------------------------------------------------- #
# http
# --------------------------------------------------------------------------- #
class ApiError(Exception):
    def __init__(self, status: int, payload):
        self.status = status
        self.payload = payload
        super().__init__(f"HTTP {status}: {payload}")


def _request(method: str, url: str, *, token: str = "", body=None, timeout: float = 70.0):
    data = None
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8") or "{}"
            return json.loads(raw)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            payload = json.loads(raw)
        except ValueError:
            payload = {"error": raw[:400] or exc.reason}
        raise ApiError(exc.code, payload)
    except urllib.error.URLError as exc:
        raise ApiError(0, {"error": "network error", "reason": str(exc.reason)})


def _die(msg: str, code: int = 1):
    print("technode: " + msg, file=sys.stderr)
    raise SystemExit(code)


def _require_key() -> str:
    key = _api_key()
    if not key:
        _die("no API key. Run `technode login` or set TECHNODE_API_KEY.\n"
             "        Get a key at https://technode.network/developers")
    return key


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_login(args) -> int:
    key = (args.key or "").strip()
    if not key:
        try:
            key = getpass.getpass("TechNode API key (tn_test_… / tn_live_…): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 1
    if not key:
        _die("no key provided.")
    if not (key.startswith("tn_test_") or key.startswith("tn_live_")):
        print("technode: warning — key does not start with tn_test_/tn_live_; saving anyway.",
              file=sys.stderr)
    cfg = _load_config()
    cfg["api_key"] = key
    _save_config(cfg)
    print(f"Saved key to {CONFIG_PATH} (chmod 600).")
    print("Try:  technode models   then   technode infer \"hello\"")
    return 0


def cmd_logout(args) -> int:
    cfg = _load_config()
    if "api_key" in cfg:
        del cfg["api_key"]
        _save_config(cfg)
        print("Removed saved API key.")
    else:
        print("No saved API key.")
    return 0


def cmd_models(args) -> int:
    base = _base_url()
    data = None
    for url in (base + "/api/models", CATALOG_FALLBACK_URL):
        try:
            data = _request("GET", url, timeout=15)
            break
        except ApiError:
            continue
    if data is None:
        _die("could not reach the model catalog.")
    models = data.get("models", data) if isinstance(data, dict) else data
    if not models:
        print("No models currently available.")
        return 0
    if args.json:
        print(json.dumps(models, indent=2, ensure_ascii=False))
        return 0
    print(f"{'MODEL':22} {'QUANT':8} {'ROLE':22} {'STATUS':10} AVAIL")
    print("-" * 72)
    for m in models:
        role = m.get("role")
        role = ",".join(role) if isinstance(role, list) else str(role or "")
        avail = "yes" if m.get("available", m.get("serving")) else "no"
        warm = " (warm)" if m.get("warm") else ""
        print(f"{str(m.get('id','')):22} {str(m.get('quant','')):8} "
              f"{role[:22]:22} {str(m.get('status','')):10} {avail}{warm}")
    return 0


def cmd_infer(args) -> int:
    key = _require_key()
    prompt = args.prompt
    if prompt == "-" or (prompt is None and not sys.stdin.isatty()):
        prompt = sys.stdin.read()
    if not prompt or not prompt.strip():
        _die("empty prompt.")
    body = {"prompt": prompt}
    if args.model:
        body["model"] = args.model
    if args.max_tokens is not None:
        body["max_new_tokens"] = args.max_tokens
    if args.temperature is not None:
        body["temperature"] = args.temperature
    url = _base_url() + "/api/dispatch"
    t0 = time.time()
    try:
        res = _request("POST", url, token=key, body=body)
    except ApiError as exc:
        return _handle_api_error(exc)
    if args.json:
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0
    text = res.get("generated") or res.get("text") or ""
    print(text.rstrip("\n"))
    if not args.quiet:
        ms = res.get("e2e_ms") or int((time.time() - t0) * 1000)
        meta = [f"model={res.get('model', args.model or '?')}"]
        if res.get("device"):
            meta.append(f"device={res['device']}")
        meta.append(f"{ms}ms")
        if res.get("cost_krw") is not None:
            meta.append(f"cost=₩{res['cost_krw']}")
        if res.get("balance_after_krw") is not None:
            meta.append(f"balance=₩{res['balance_after_krw']}")
        print("  · " + "  ".join(meta), file=sys.stderr)
    return 0


def _handle_api_error(exc: ApiError) -> int:
    p = exc.payload if isinstance(exc.payload, dict) else {"error": exc.payload}
    err = p.get("error", "request failed")
    reason = p.get("reason") or p.get("hint")
    if exc.status == 401:
        _die(f"unauthorized — {reason or err}.\n"
             "        Check your key:  technode login")
    if exc.status == 402:
        _die(f"out of credit — {p.get('hint') or err}.\n"
             "        Top up at https://technode.network/app/customer")
    if exc.status == 0:
        _die(f"network error — {reason or err}.")
    msg = f"{err}" + (f" ({reason})" if reason else "")
    _die(f"server returned {exc.status}: {msg}")
    return 1  # unreachable


def _fmt_lease(res: dict) -> None:
    rid = res.get("lease_id", "?")
    gpu = res.get("gpu_name") or res.get("provider_id") or "?"
    jhref = res.get("lab_join_url") or res.get("lab_url") or ""
    token = res.get("lab_token") or ""
    rem = res.get("remaining_s")
    print("GPU leased ✓")
    print(f"  lease id:  {rid}")
    print(f"  gpu:       {gpu}")
    if jhref:
        print(f"  jupyter:   {jhref}")
    if token:
        print(f"  token:     {token}")
    if rem is not None:
        print(f"  expires:   in {int(rem) // 60} min")
    print(f"\nStatus:   technode gpu status {rid}")
    print(f"Release:  technode gpu release {rid}")


def cmd_gpu_lease(args) -> int:
    key = _require_key()
    body = {}
    if args.gpu:
        body["gpu_id"] = args.gpu
    try:
        res = _request("POST", _base_url() + "/api/lease", token=key, body=body, timeout=30)
    except ApiError as exc:
        if exc.status == 409:
            p = exc.payload if isinstance(exc.payload, dict) else {}
            _die(f"no free GPU slot right now — {p.get('error', 'all slots in use')}.\n"
                 "        Try again shortly (`technode gpu list` shows active leases).")
        return _handle_api_error(exc)
    if args.json:
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0
    _fmt_lease(res)
    return 0


def cmd_gpu_status(args) -> int:
    try:
        res = _request("GET", _base_url() + "/api/lease?id=" + args.lease_id, timeout=15)
    except ApiError as exc:
        return _handle_api_error(exc)
    if args.json:
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0
    rem = res.get("remaining_s")
    state = res.get("status") or ("active" if rem else "unknown")
    print(f"lease {args.lease_id}: {state}"
          + (f" · {int(rem) // 60} min left" if rem is not None else ""))
    return 0


def cmd_gpu_release(args) -> int:
    try:
        res = _request("DELETE", _base_url() + "/api/lease?id=" + args.lease_id, timeout=15)
    except ApiError as exc:
        return _handle_api_error(exc)
    if args.json:
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0
    print(f"Released lease {args.lease_id}.")
    return 0


def cmd_gpu_list(args) -> int:
    try:
        res = _request("GET", _base_url() + "/api/leases", timeout=15)
    except ApiError as exc:
        return _handle_api_error(exc)
    leases = res.get("leases", res) if isinstance(res, dict) else res
    if args.json:
        print(json.dumps(leases, indent=2, ensure_ascii=False))
        return 0
    if not leases:
        print("No active GPU leases.")
        return 0
    print(f"{'LEASE ID':40} {'GPU':20} REMAINING")
    print("-" * 72)
    for l in leases:
        rem = l.get("remaining_s")
        print(f"{str(l.get('lease_id','')):40} {str(l.get('gpu_name','')):20} "
              + (f"{int(rem) // 60} min" if rem is not None else "?"))
    return 0


def cmd_whoami(args) -> int:
    key = _api_key()
    base = _base_url()
    if not key:
        print("Not logged in. Run `technode login`.")
        print(f"Endpoint: {base}")
        return 1
    masked = key[:12] + "…" + key[-4:] if len(key) > 18 else key
    src = "env TECHNODE_API_KEY" if os.environ.get("TECHNODE_API_KEY") else CONFIG_PATH
    print(f"Key:      {masked}  (from {src})")
    print(f"Endpoint: {base}")
    return 0


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="technode",
        description="Run inference on TechNode's compressed open-model GPU grid.",
    )
    p.add_argument("--version", action="version",
                   version=f"technode {__version__}")
    sub = p.add_subparsers(dest="command")

    sp = sub.add_parser("login", help="save your API key locally")
    sp.add_argument("key", nargs="?", help="tn_test_… key (prompted if omitted)")
    sp.set_defaults(func=cmd_login)

    sp = sub.add_parser("logout", help="remove the saved API key")
    sp.set_defaults(func=cmd_logout)

    sp = sub.add_parser("models", help="list the available compressed models")
    sp.add_argument("--json", action="store_true", help="raw JSON output")
    sp.set_defaults(func=cmd_models)

    sp = sub.add_parser("infer", help="run a text generation request")
    sp.add_argument("prompt", nargs="?", help="prompt text ('-' or piped stdin to read stdin)")
    sp.add_argument("-m", "--model", help="model id (see `technode models`)")
    sp.add_argument("-n", "--max-tokens", type=int, dest="max_tokens",
                    help="max new tokens")
    sp.add_argument("-t", "--temperature", type=float, help="sampling temperature")
    sp.add_argument("--json", action="store_true", help="raw JSON output")
    sp.add_argument("-q", "--quiet", action="store_true",
                    help="suppress the timing/cost footer")
    sp.set_defaults(func=cmd_infer)

    sp = sub.add_parser("whoami", help="show the active key + endpoint")
    sp.set_defaults(func=cmd_whoami)

    # gpu: rent a whole GPU (Jupyter session), separate from `infer`.
    gpu = sub.add_parser("gpu", help="rent a GPU (Jupyter lab session)")
    gsub = gpu.add_subparsers(dest="gpu_command")

    g = gsub.add_parser("lease", help="lease a GPU slot (~60 min)")
    g.add_argument("--gpu", help="GPU id hint (optional)")
    g.add_argument("--json", action="store_true", help="raw JSON output")
    g.set_defaults(func=cmd_gpu_lease)

    g = gsub.add_parser("status", help="check a lease")
    g.add_argument("lease_id")
    g.add_argument("--json", action="store_true", help="raw JSON output")
    g.set_defaults(func=cmd_gpu_status)

    g = gsub.add_parser("release", help="release a lease early")
    g.add_argument("lease_id")
    g.add_argument("--json", action="store_true", help="raw JSON output")
    g.set_defaults(func=cmd_gpu_release)

    g = gsub.add_parser("list", help="list active GPU leases")
    g.add_argument("--json", action="store_true", help="raw JSON output")
    g.set_defaults(func=cmd_gpu_list)

    gpu.set_defaults(func=lambda a: (gpu.print_help() or 0))

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print()
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
