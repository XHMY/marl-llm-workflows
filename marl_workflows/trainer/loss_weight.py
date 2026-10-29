"""Per-role loss weight for the shared-policy (SP) update (control T1).

Enabled with ``+trainer.loss_weight_per_role.mode=length`` (or ``static`` with
``.weights.<role>``). Exact port of the research fork's
``AgentWorkflowPPOTrainer._apply_per_role_loss_weight``
(``rllm/trainer/verl/agent_workflow_trainer.py:185-250``); roles are parsed with
:func:`marl_workflows.contracts.roles_from_trajectory_ids`, which is the same rule.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from marl_workflows.contracts import LOSS_WEIGHT_PREFIX, roles_from_trajectory_ids

__all__ = ["LOSS_WEIGHT_MODES", "apply_per_role_loss_weight"]

LOSS_WEIGHT_MODES = ("length", "static")


def apply_per_role_loss_weight(batch: Any, cfg: Any) -> tuple[Any, dict[str, float]]:
    """Scale each row's advantages by its role's weight; return ``(batch, metrics)``.

    Row weights are normalised so that their mean over rows with a non-empty response is 1;
    rows with an empty response get weight 1 and are excluded from every mean. Because the
    policy loss is linear in the advantage, scaling the advantage is an exact per-row loss
    weight. ``returns`` is left untouched.

    - ``length``: w_role = mean response length of the role / mean over all rows.
    - ``static``: w_role = ``cfg.weights.<role>`` (missing roles get 1).
    """
    mode = str(cfg.get("mode", "none"))
    if mode not in LOSS_WEIGHT_MODES:
        raise ValueError(f"loss_weight_per_role.mode must be one of none/length/static, got {mode!r}")
    roles = np.array(roles_from_trajectory_ids(batch.non_tensor_batch["trajectory_ids"]), dtype=object)
    lengths = batch.batch["response_mask"].sum(-1).float().cpu()
    valid = lengths > 0
    any_valid = bool(valid.any())
    role_names = sorted(set(roles.tolist()))
    overall_mean = lengths[valid].mean().item() if any_valid else 0.0
    static_weights = (cfg.get("weights", None) or {}) if mode == "static" else {}

    raw: dict[str, float] = {}
    role_len: dict[str, float] = {}
    role_rows: dict[str, int] = {}
    for role in role_names:
        in_role = torch.from_numpy(roles == role) & valid
        role_rows[role] = int(in_role.sum().item())
        role_len[role] = lengths[in_role].mean().item() if role_rows[role] > 0 else 0.0
        if mode == "length":
            raw[role] = role_len[role] / overall_mean if overall_mean > 0 and role_rows[role] > 0 else 1.0
        else:
            raw[role] = float(static_weights.get(role, 1.0))

    w = torch.ones(len(roles), dtype=torch.float64)
    for role in role_names:
        w[torch.from_numpy(roles == role)] = raw[role]
    w[~valid] = 1.0
    norm = 0.0
    if any_valid:
        norm = w[valid].mean().item()
        if norm > 0:
            w[valid] = w[valid] / norm
    norm_role = {role: (raw[role] / norm if any_valid and norm > 0 else raw[role]) for role in role_names}

    adv = batch.batch["advantages"]
    batch.batch["advantages"] = adv * w.to(device=adv.device, dtype=adv.dtype)[:, None]

    metrics: dict[str, float] = {}
    for role in role_names:
        metrics[f"{LOSS_WEIGHT_PREFIX}{role}"] = norm_role[role]
        metrics[f"{LOSS_WEIGHT_PREFIX}{role}_mean_resp_len"] = role_len[role]
        metrics[f"{LOSS_WEIGHT_PREFIX}{role}_rows"] = role_rows[role]
    metrics[f"{LOSS_WEIGHT_PREFIX}row_mean"] = w[valid].mean().item() if any_valid else 1.0
    parts = " ".join(
        f"{role}={norm_role[role]:.4g} (len {role_len[role]:.0f}, {role_rows[role]} rows)" for role in role_names
    )
    print(
        f"[loss_weight_per_role] mode={mode} {parts} row_mean={metrics[f'{LOSS_WEIGHT_PREFIX}row_mean']:.6f}",
        flush=True,
    )
    return batch, metrics
