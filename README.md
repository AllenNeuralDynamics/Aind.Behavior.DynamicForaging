# aind-dynamic-foraging

![CI](https://github.com/AllenNeuralDynamics/Aind.Behavior.DynamicForaging/actions/workflows/dynamic-foraging-cicd.yml/badge.svg)
[![PyPI - Version](https://img.shields.io/pypi/v/aind-behavior-dynamic-foraging)](https://pypi.org/project/aind-behavior-dynamic-foraging/)
[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

A repository for the Dynamic Foraging task and its associated curricula.

---

## 📋 General instructions

This repository follows the project structure laid out in the [Aind.Behavior.Services repository](https://github.com/AllenNeuralDynamics/Aind.Behavior.Services).

---

## 🔧 Prerequisites

[Pre-requisites for running the project can be found here](https://allenneuraldynamics.github.io/Aind.Behavior.Services/articles/requirements.html).

---

## 🚀 Deployment

For convenience, once third-party dependencies are installed, `Bonsai` and `python` virtual environments can be bootstrapped by running:

```powershell
./scripts/deploy.ps1
```

from the root of the repository.

## ⚙️ Generating settings files

The Dynamic Foraging task is instantiated by a set of three settings files that strictly follow a DSL schema. These files are:

- `task_logic.json`
- `rig.json`
- `session.json`

Examples on how to generate these files can be found in the `./Examples` directory of the repository. Once generated, these are the the only required inputs to run the Bonsai workflow in `./src/main.bonsai`.

The workflow can thus be executed using the [Bonsai CLI](https://bonsai-rx.org/docs/articles/cli.html):

```powershell
"./bonsai/bonsai.exe" "./src/main.bonsai" -p SessionPath=<path-to-session.json> -p RigPath=<path-to-rig.json> -p TaskLogicPath=<path-to-task_logic.json>
```

However, for a better experiment management user experience, it is recommended to use the provided experiment launcher below.

## [> ] CLI tools

### Task CLI

The platform exposes a few CLI tools to facilitate various tasks. Tools are available via:

```powershell
uv run dynamic-foraging <subcommand>
```

for a list of all sub commands available:

```powershell
uv run dynamic-foraging -h
```

You may need to install optional dependencies depending on the sub-commands you run.

### Curriculum CLI

Curricula are available via the `curriculum` CLI entry point. For a full list of commands:

```powershell
uv run curriculum -h
```

#### `list` - List Available Curricula

```bash
uv run curriculum list
```

#### `init` - Initialize a Curriculum

Creates an initial trainer state for enrolling a subject in a curriculum.

```bash
# Start at the first stage
uv run curriculum init --curriculum coupled_baiting --output initial_state.json

# Start at a specific stage
uv run curriculum init --curriculum coupled_baiting --stage s_stage_1 --output initial_state.json
```

#### `run` - Run a Curriculum

Evaluates a curriculum based on session data and current trainer state.

```bash
uv run curriculum run \
  --data-directory /path/to/session/data \
  --input-trainer-state current_state.json \
  --output-suggestion /path/to/output
```

Force a specific curriculum:

```bash
uv run curriculum run \
  --data-directory /path/to/session/data \
  --input-trainer-state current_state.json \
  --curriculum coupled_baiting \
  --output-suggestion /path/to/output
```

#### `version` / `dsl-version` - Show Versions

```bash
uv run curriculum version      # Package version
uv run curriculum dsl-version  # Underlying DSL library version
```

---

## Typical curriculum workflow

1. **List available curricula:**
   ```bash
   uv run curriculum list
   ```

2. **Initialize a subject:**
   ```bash
   uv run curriculum init --curriculum coupled_baiting --output trainer_state.json
   ```

3. **After a session, evaluate progress:**
   ```bash
   uv run curriculum run \
     --data-directory /path/to/session/data \
     --input-trainer-state trainer_state.json \
     --output-suggestion /path/to/output
   ```

4. **Use the suggestion for the next session:**
   The `suggestion.json` output can be passed as `--input-trainer-state` for the next session.

---

## Style guide

To keep things clear, the following naming conventions are recommended:

- **Policies** should start with `p_` (e.g., `p_identity_policy`)
- **Policy transitions** should start with `pt_`
- **Stages** should start with `s_` (e.g., `s_stage1`)
- **Stage transitions** should start with `st_` and be named after the stages they transition between (e.g., `st_s_stage1_s_stage2`)

Define the following modules within a curriculum:

- **metrics**: Defines (or imports) metrics classes and how to calculate them from data
- **stages**: Defines the different stages of the task, including task settings and optionally policies
- **curriculum**: Defines transitions between stages and generates the entry point to the application

---

## 🎮 Experiment launcher (temporarily CLABE)

To manage experiments and input files, this repository contains a launcher script that can be used to run the Dynamic Foraging task. A default script is located at `./scripts/aind-launcher.py`. It can be run from the command line as follows:

```powershell
uv run clabe run ./scripts/aind-launcher.py
# or uv run ./scripts/main.py
```

Additional arguments can be passed to the script as needed. For instance to allow the script to run with uncommitted changes in the repository, the `--allow-dirty` flag can be used:

```powershell
uv run clabe run ./scripts/aind-launcher.py --allow-dirty
```

or via a `./local/clabe.yml` file. Additional custom launcher scripts can be created and used as needed. See documentation in the [`clabe` repository](https://allenneuraldynamics.github.io/clabe/) for more details.


## 🔍 Primary data quality-control

Once an experiment is collected, the primary data quality-control script can be run to check the data for issues. This script can be launcher using:

```powershell
uv run dynamic-foraging data-qc <path-to-data-dir>
```

## 🔄 Regenerating schemas

DSL schemas can be modified in `./src/aind_behavior_dynamic_foraging/rig.py` (or `(...)/task_logic`.py`).

Once modified, changes to the DSL must be propagated to `json-schema` and `csharp` API. This can be done by running:

```powershell
uv run dynamic-foraging regenerate
```
