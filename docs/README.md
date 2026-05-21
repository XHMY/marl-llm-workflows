# Documentation

This directory contains the public-facing documentation for the multi-agent
RL extensions that this fork adds on top of [rllm
v0.2.1](https://github.com/rllm-org/rllm). The MkDocs site is configured by
the top-level `mkdocs.yml`.

## Building locally

```bash
pip install mkdocs mkdocs-material 'mkdocstrings[python]' mkdocs-autorefs pymdown-extensions
mkdocs serve            # live preview at http://localhost:8000
mkdocs build            # static site under ./site
```

## What lives here

- `multi-agent/` — the multi-agent RL documentation set (overview, the three
  workflows + single-agent baseline, isolated- vs shared-policy design,
  multi-agent LoRA implementation, training loop, trajectory dumps and
  evaluation, monitoring, end-to-end experiment recipe, intervention notes).
- `assets/`, `stylesheets/` — images and styling consumed by the MkDocs theme.

## Upstream rllm documentation

This release ships only the multi-agent docs. For everything else
(installation alternatives, the rllm SDK, core concepts, the upstream
example library, API reference), read the corresponding pages in upstream
rllm at:

<https://github.com/rllm-org/rllm/tree/v0.2.1/docs>
