# Aind.Behavior.DynamicForaging.Curricula

![CI](https://github.com/AllenNeuralDynamics/Aind.Behavior.DynamicForaging.Curricula/actions/workflows/dynamic-foraging-curricula.yml/badge.svg)
[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)


A repository of curricula for [Dynamic foraging task](https://github.com/AllenNeuralDynamics/Aind.Behavior.DynamicForaging).

## CLI Reference

Curricula are modules of the main package: `aind_behavior_dynamic_foraging_curricula.<curriculum_name>`. 

All curricula are available via the `curriculum` CLI entry point. The following commands are available:

### Getting Help

Display all available commands:

```bash
uv run curriculum -h
```

Get help for a specific command:

```bash
uv run curriculum <command> -h
```

### `list` - List Available Curricula

Lists all curricula available in this repository.

**Usage:**

```bash
uv run curriculum list
```

**Example output:**
```
Available curricula:
  coupled_baiting
```

### `init` - Initialize a Curriculum

Creates an initial trainer state for enrolling a subject in a curriculum. This generates the starting point for curriculum execution.

**Required Arguments:**
- `--curriculum <name>`: The curriculum to enroll in (required)

**Optional Arguments:**
- `--output <path>`: Path to save the enrollment trainer state as a JSON file
- `--stage <name>`: If provided, enroll at a specific stage instead of the first stage

**Examples:**

Initialize the depletion curriculum (starts at first stage):

```bash
uv run curriculum init --curriculum coupled_baiting --output initial_state.json
```

Initialize at a specific stage:

```bash
uv run curriculum init --curriculum coupled_baiting --stage s_stage_1 --output initial_state.json
```

Print to stdout without saving:

```bash
uv run curriculum init --curriculum coupled_baiting
```

### `run` - Run a Curriculum

Evaluates a curriculum based on session data and the current trainer state, producing a suggestion for the next stage.

**Required Arguments:**
- `--data-directory <path>`: Path to the session data directory for calculating metrics
- `--input-trainer-state <path>`: Path to the current trainer state JSON file

**Optional Arguments:**
- `--stage_changed <bool>`: Flag to indicate whether stage in previous session differs from current session. Default is false.
- `--previous_metrics <path>`: Path to the previous session metrics JSON file
- `--curriculum <name>`: Forces the use of a specific curriculum, bypassing automatic detection
- `--output-suggestion <path>`: Directory path to save the suggestion as `suggestion.json`
- `--mute-suggestion`: Disables printing the suggestion to stdout (useful when only saving to file)

**Examples:**

Run curriculum with automatic detection:

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
  --curriculum depletion \
  --output-suggestion /path/to/output
```

Run without saving (print to stdout only):

```bash
uv run curriculum run \
  --data-directory /path/to/session/data \
  --input-trainer-state current_state.json
```

Run and save without printing:

```bash
uv run curriculum run \
  --data-directory /path/to/session/data \
  --input-trainer-state current_state.json \
  --output-suggestion /path/to/output \
  --mute-suggestion
```

Quick demo with template curriculum:

```bash
uv run curriculum run \
  --data-directory "demo" \
  --input-trainer-state "foo.json" \
  --curriculum "template"
```

### `version` - Show Package Version

Displays the version of this package.

**Usage:**

```bash
uv run curriculum version
```

**Example output:**
```
0.2.0
```

### `dsl-version` - Show DSL Version

Displays the version of the underlying `aind-behavior-curriculum` DSL library.

**Usage:**

```bash
uv run curriculum dsl-version
```

**Example output:**
```
0.0.37
```

## Typical Workflow

1. **List available curricula:**
   ```bash
   uv run curriculum list
   ```

2. **Initialize a subject in a curriculum:**
   ```bash
   uv run curriculum init --curriculum depletion --output trainer_state.json
   ```

3. **After a training session, evaluate progress:**
   ```bash
   uv run curriculum run \
     --data-directory /path/to/session/data \
     --input-trainer-state trainer_state.json \
     --output-suggestion /path/to/output
   ```

4. **Use the suggestion output for the next session:**
   The `suggestion.json` file contains the updated trainer state and can be used as `--input-trainer-state` for the next session.


## Style guide

To keep things clear, I suggest the following naming convention:

* **Policies** should start with `p_` (e.g., `p_identity_policy`)
* **Policy transitions** should start with `pt_`
* **Stages** should start with `s_` (e.g., `s_stage1`)
* **Stage transitions** should start with `st_` and should be named after the stages they transition between (e.g., `st_s_stage1_s_stage2`)

Define the following modules:

* **metrics**: Defines (or imports) metrics classes and how to calculate them from data
* **stages**: Defines the different stages of the Dynamic foraging task. This includes task settings and, optionally, policies
* **curriculum**: Defines the transitions between the stages and generate entry point to the application

## Contributors

Contributions to this repository are welcome! However, please ensure that your code adheres to the recommended DevOps practices below:

### Linting

We use [ruff](https://docs.astral.sh/ruff/) as our primary linting tool.

### Testing

Attempt to add tests when new features are added.
To run the currently available tests, run `uv run pytest` from the root of the repository.

### Lock files

We use [uv](https://docs.astral.sh/uv/) to manage our lock files and therefore encourage everyone to use uv as a package manager as well.