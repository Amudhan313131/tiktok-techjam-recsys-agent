# How to run REX with Docker

This guide takes you from a fresh checkout to a real KuaiRand-Pure research
run. You do not need to install the Python packages or LightGBM on your
computer; the Docker image contains the locked runtime.

The run is validation-only. It can stop early when the search converges, and it
will never run for longer than six hours. It does not create or score hidden
test predictions.

## Before you start

Install these programs:

- Git;
- Python 3.11 or newer;
- Docker Desktop on macOS or Windows, or Docker Engine with Buildx on Linux.

Start Docker before continuing. On Windows, the simplest supported setup is
Docker Desktop with WSL2; run the commands below inside your WSL terminal.

## 1. Pull the latest code

For a new checkout:

```bash
git clone https://github.com/Amudhan313131/tiktok-techjam-recsys-agent.git
cd tiktok-techjam-recsys-agent
```

If you already have the repository:

```bash
cd tiktok-techjam-recsys-agent
git pull --ff-only origin main
```

The production image must be built from a clean, committed checkout. This
command should print nothing:

```bash
git status --short
```

## 2. Put KuaiRand-Pure in the data folder

Download and extract the official archive:

```bash
mkdir -p data
curl -L https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz \
  -o data/KuaiRand-Pure.tar.gz
tar -xzf data/KuaiRand-Pure.tar.gz -C data
```

You can also download and extract it manually. In either case, these files must
exist at these exact paths:

```text
data/KuaiRand-Pure/data/log_standard_4_08_to_4_21_pure.csv
data/KuaiRand-Pure/data/log_standard_4_22_to_5_08_pure.csv
data/KuaiRand-Pure/data/video_features_basic_pure.csv
```

The archive contains additional files. Leave them in the same directory. REX
checks the required file hashes, dates, row identities, and validation/test
separation before research begins.

## 3. Choose one LLM login method

The LLM acts as the researcher: it proposes a constrained model-code change and
explains the result. The trained FM or LightGBM model makes the video
predictions.

### Option A: Codex CLI

Install Codex on macOS, Linux, or WSL:

```bash
curl -fsSL https://chatgpt.com/codex/install.sh | sh
codex
```

The first time `codex` opens, choose **Sign in with ChatGPT** and finish the
login. REX later mounts your `~/.codex` login directory read-only into the
trusted Docker controller. See the [official Codex CLI
guide](https://learn.chatgpt.com/docs/codex/cli) if installation or login fails.

### Option B: Claude CLI

Install Claude Code, then open it once and finish its login:

```bash
npm install -g @anthropic-ai/claude-code
claude
```

REX later mounts your `~/.claude` login directory read-only into the trusted
Docker controller. See the [official Claude Code setup
guide](https://docs.anthropic.com/en/docs/claude-code/getting-started) if needed.

### Option C: OpenAI API key

Create an API key using the [official OpenAI API
quickstart](https://developers.openai.com/api/docs/quickstart), then set both
variables in the same terminal that will start REX:

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_MODEL="your-authorized-model-id"
```

API mode can incur charges. Selecting `--llm openai_api` is treated as explicit
authorization for REX's bounded paid API calls. Never save the real key in the
repository, Dockerfile, or a committed `.env` file.

## 4. Build the Docker image and run

Build the image. The script automatically selects Intel/AMD or ARM/Apple
Silicon and refuses to build if the Git checkout is dirty.

```bash
scripts/build_docker.sh --tag rex:local
```

Create a unique name and an output directory outside the repository:

```bash
RUN_ID="rex-$(date +%Y%m%d-%H%M%S)"
RUN_DIR="$HOME/rex-runs/$RUN_ID"
echo "Run ID: $RUN_ID"
echo "Output: $RUN_DIR"
```

Then start exactly one of the following commands.

### Run with Codex

```bash
python3 scripts/run_docker_rehearsal.py start \
  --source-root "$PWD" \
  --data-dir "$PWD/data/KuaiRand-Pure/data" \
  --output-dir "$RUN_DIR" \
  --run-id "$RUN_ID" \
  --image rex:local \
  --llm codex_cli \
  --codex-home "$HOME/.codex"
```

### Run with Claude

```bash
python3 scripts/run_docker_rehearsal.py start \
  --source-root "$PWD" \
  --data-dir "$PWD/data/KuaiRand-Pure/data" \
  --output-dir "$RUN_DIR" \
  --run-id "$RUN_ID" \
  --image rex:local \
  --llm claude_cli \
  --claude-home "$HOME/.claude"
```

### Run with an OpenAI API key

```bash
python3 scripts/run_docker_rehearsal.py start \
  --source-root "$PWD" \
  --data-dir "$PWD/data/KuaiRand-Pure/data" \
  --output-dir "$RUN_DIR" \
  --run-id "$RUN_ID" \
  --image rex:local \
  --llm openai_api
```

Keep that terminal open. The launcher verifies the source, data, image,
LightGBM, and live LLM; starts the autonomous research loop; injects one
controlled controller failure; resumes safely; and seals the evidence.

## Check progress from another terminal

Copy the output path printed by the first terminal and use it in place of
`<your-run-id>`:

```bash
python3 scripts/run_docker_rehearsal.py status \
  --output-dir "$HOME/rex-runs/<your-run-id>"
```

This is read-only. While the run is active, do not edit the source, rebuild the
image, restart the run, or create test predictions.

## What success looks like

The status `phase` changes to `complete`, and the output directory contains:

- `docker_r3_manifest.json` — the sealed rehearsal summary;
- `<run-id>/best-valid/` — the immutable validation champion;
- `<run-id>/report/` — per-iteration hypotheses, diffs, metrics, failures,
  resource usage, and manual-intervention summary;
- `controller-logs/` — controller and recovery logs.

Final submission generation is a separate, explicitly authorized step. See
the [final submission section](../README.md#final-submission-generation-and-handoff)
after the validation run has completed successfully.

## Common problems

| Message or symptom | What to do |
|---|---|
| Docker cannot be reached | Start Docker Desktop or the Docker Engine, then retry. |
| The checkout is dirty | Commit the intended source changes first. Do not delete work just to make the check pass. |
| A data file is missing or has the wrong hash | Re-extract the official KuaiRand-Pure archive into `data/KuaiRand-Pure/`. |
| Codex or Claude authentication fails | Open `codex` or `claude` on the host, finish login, then pass the matching home-directory flag. |
| API authentication fails | Set `OPENAI_API_KEY` and `OPENAI_MODEL` in the terminal that starts the run. |
| The output directory already contains a run | Create a new `RUN_ID` and `RUN_DIR`; never overwrite a previous run. |

For security design, image provenance, direct controller commands, and release
details, see [Docker production details](docker-production.md).
