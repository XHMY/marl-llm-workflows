# Source this file to use the repository without installing it:
#   source scripts/env.sh
# It puts this package and the two pinned submodules first on PYTHONPATH, so they take
# precedence over any rllm/verl already installed in the active environment.
# The alternative is editable installs in the order verl, rllm, this package (docs/design.md).

if [ -n "${BASH_SOURCE[0]:-}" ]; then
    _marl_env_file="${BASH_SOURCE[0]}"
elif [ -n "${ZSH_VERSION:-}" ]; then
    # zsh: %x expands to the file being sourced.
    _marl_env_file="${(%):-%x}"
else
    _marl_env_file="$0"
fi
MARL_REPO_ROOT="$(cd "$(dirname "${_marl_env_file}")/.." && pwd)"
export MARL_REPO_ROOT
export PYTHONPATH="${MARL_REPO_ROOT}:${MARL_REPO_ROOT}/third_party/rllm:${MARL_REPO_ROOT}/third_party/verl${PYTHONPATH:+:${PYTHONPATH}}"
unset _marl_env_file
