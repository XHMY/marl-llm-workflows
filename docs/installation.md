# Installation

Requirements: Linux, Python 3.11, CUDA 12.8 drivers, and GPUs with at least 48 GB of memory for
training (80 GB for Qwen3-4B). Evaluation of the small models runs on one 24 GB GPU.

The repository pins its two upstream dependencies as git submodules and never modifies them:

| Submodule | Upstream | Pin |
|---|---|---|
| `third_party/rllm` | https://github.com/rllm-org/rllm | `186d0bbbb5bfe18f8f6b8e4eac6db9379cf27738` |
| `third_party/verl` | https://github.com/volcengine/verl | tag `v0.6.1` (`d62da4950573d7a4b7ef2362337952e7ab59e78d`) |

```bash
git clone --recurse-submodules <repository-url> marl-llm-workflows
cd marl-llm-workflows
git submodule status            # shows the two pins above
```

## 1. The Python environment

```bash
conda create -n marl python=3.11 -y
conda activate marl
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements/stack.txt     # vLLM 0.11.0, transformers 4.56.1, peft 0.18.1, ray 2.54.0, ...
```

`flash-attn==2.8.3` builds against the installed torch; if pip tries to build it in an isolated
environment, install it on its own with `pip install flash-attn==2.8.3 --no-build-isolation`.
`requirements/lock-fork-env.txt` is the full `pip freeze` of the environment that produced the paper
runs, for reference.

## 2. Put the package and the submodules on the import path

Choose one of the two ways.

**A. No install.** Source the environment script in every shell (and in job scripts):

```bash
source scripts/env.sh
python -c "import verl, rllm; print(verl.__file__, rllm.__file__)"   # both inside third_party/
```

`scripts/env.sh` prepends the repository and both submodules to `PYTHONPATH`, so they take
precedence over any rllm or verl installed in the environment.

**B. Editable installs**, in this order:

```bash
pip install --no-deps -e third_party/verl
pip install --no-deps -e third_party/rllm   # --no-deps: rllm's verl extra would reinstall verl from PyPI
pip install -e ".[dev]"
```

This also installs the console scripts `marl-train`, `marl-eval` and `marl-launch`. SLURM rendering
(`scripts/launch.py --slurm-config`) reads its templates from the checkout, so keep the clone.

## 3. Data location

Prepared datasets are written under `$MARL_DATA_DIR` (default `~/.cache/marl-llm-workflows`), never
inside the submodules:

```bash
export MARL_DATA_DIR=$PWD/data_cache
python -m marl_workflows.data.prepare_smoke    # the 40-problem fixture used by the smoke test
```

The full datasets are prepared as described in [reproducing.md](reproducing.md#1-prepare-the-data).

## 4. Check the installation

```bash
pytest tests/ -q                 # CPU tests (GPU tests are deselected by default)
pytest -m gpu tests/gpu -q       # on a GPU node
bash scripts/smoke_gpu.sh        # end-to-end: two training steps of Voting with Qwen3-0.6B on one GPU
```

The smoke test trains Voting with one adapter per role and then with a shared adapter, two steps
each, and checks the per-role metrics, the saved adapters, the checkpoint metadata and that a
second launch of a finished run exits immediately. Outputs go to `outputs/smoke/`; set
`SMOKE_EVAL=1` to also run the evaluator on the IP checkpoints.
