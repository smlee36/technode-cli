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
