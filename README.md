# technode

Run inference on **TechNode** — a GPU grid serving *compressed* open models
(Qwen, Granite, gpt-oss, Devstral, Gemma, EXAONE…) at consumer-GPU prices.

```bash
pip install technode-cli       # the command is `technode`
technode login                 # paste your tn_test_… key (or set TECHNODE_API_KEY)
technode models                # list the compressed catalog
technode infer "Explain quantization in one line." --model qwen2.5-7b
```

Zero dependencies — pure Python stdlib, runs anywhere Python ≥3.8 does.

## Commands

| Command | What it does |
|---|---|
| `technode login [key]` | Save your API key to `~/.technode/config.json` (chmod 600). |
| `technode logout` | Remove the saved key. |
| `technode models [--json]` | List available models (id, quantization, role). |
| `technode infer PROMPT [-m MODEL] [-n MAX_TOKENS] [-t TEMP] [--json] [-q]` | Text generation. `-` or piped stdin reads the prompt from stdin. |
| `technode whoami` | Show the active key (masked) + endpoint. |
| `technode gpu lease/list/status/release` | Rent a whole GPU (Jupyter lab session). |

## Become a provider (share your GPU)

Got an NVIDIA Linux box? Join the grid and serve models — **outbound-only, works
behind any NAT** (no Tailscale, no inbound ports):

```bash
technode provider register --gpu "RTX 4090" --vram 24
technode provider serve --llama-server /path/to/llama-server   # pull-mode worker (llama.cpp)
technode provider status
```

### Data-center / IDC GPUs (B200·B300·H100, multi-GPU)

For datacenter GPU nodes, use the **vLLM backend** with tensor-parallel across GPUs:

```bash
pip install -U technode-cli vllm        # vLLM needs CUDA GPUs + drivers
technode provider register --gpu "8x B200" --vram 1440
technode provider serve --backend vllm --tp 8     # 8-GPU tensor-parallel
technode provider install --backend vllm --tp 8   # → systemd unit (boot persistence)
```

One-shot onboarding (detects GPUs, installs, registers, serves):

```bash
curl -fsSL https://technode.network/idc.sh | bash -s -- --name "IDC-node-01"
```

`serve` polls the broker for jobs it can run, executes them on your GPU, and
returns the results. Needs a llama.cpp `llama-server` binary (CUDA build for
NVIDIA) and operator approval before it receives live jobs.

## Configuration

| Setting | Env var | Default |
|---|---|---|
| API key | `TECHNODE_API_KEY` | — (from `technode login`) |
| Endpoint | `TECHNODE_BASE_URL` | `https://technode.network` |

Get a key (free beta): <https://technode.network/developers>

## Examples

```bash
# pick a coder model
technode infer "Write a Python one-liner to flatten a list of lists." -m qwen2.5-coder-7b

# read the prompt from a file / pipe
cat prompt.txt | technode infer -

# machine-readable
technode infer "hi" --json
```
