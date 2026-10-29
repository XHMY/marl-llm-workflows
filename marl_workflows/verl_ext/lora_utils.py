# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# COPIED FROM verl@v0.6.1:verl/utils/fsdp_utils.py:569-666 (layered_summon_lora_params, collect_lora_params);
# diff regions marked MARL(adapter_name).
"""Adapter-aware LoRA parameter collection.

Upstream collects only the PEFT adapter named ``default`` (``get_peft_model_state_dict`` is called
without ``adapter_name``). Under IP every role has its own adapter, so both functions take an extra
``adapter_name`` argument and pass it to every ``get_peft_model_state_dict`` call. With the default
value they behave exactly like upstream. The returned keys have the adapter name stripped (PEFT
does that), e.g. ``base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight``.
"""

from collections import OrderedDict

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from verl.utils.device import get_device_name, get_torch_device
from verl.utils.fsdp_utils import fsdp_version

__all__ = ["collect_lora_params", "layered_summon_lora_params"]


# >>> MARL(adapter_name): extra argument
def layered_summon_lora_params(fsdp_module, adapter_name: str = "default") -> OrderedDict:
    # <<< MARL(adapter_name)
    from peft.utils.save_and_load import get_peft_model_state_dict

    def __prefix_submodules(module, prefix):
        for name, submodule in module.named_modules():
            if name.startswith(prefix) and "." not in name[len(prefix) :]:
                yield name, submodule

    lora_params = OrderedDict()
    prefix_list = [
        # fsdp
        "_fsdp_wrapped_module.base_model.model.",
        "_fsdp_wrapped_module.base_model.model.model.",
        "_fsdp_wrapped_module.base_model.model.model.layers.",
        "_fsdp_wrapped_module.base_model.model.model.language_model.layers.",
        # fsdp2
        "base_model.model.",
        "base_model.model.model.",
        "base_model.model.model.layers.",
        "base_model.model.model.language_model.layers.",
    ]
    peft_model = getattr(fsdp_module, "_fsdp_wrapped_module", fsdp_module)
    for prefix in prefix_list:
        for name, submodule in __prefix_submodules(fsdp_module, prefix):
            prefix = name.replace("_fsdp_wrapped_module.base_model.model.", "base_model.model.")
            if name.endswith(".model") or name.endswith(".layers"):
                continue
            if fsdp_version(submodule) > 0:
                with FSDP.summon_full_params(submodule, writeback=False):
                    # >>> MARL(adapter_name)
                    sub_lora_params = get_peft_model_state_dict(
                        peft_model, state_dict=submodule.state_dict(), adapter_name=adapter_name
                    )
                    # <<< MARL(adapter_name)
                    sub_lora_params = {
                        f"{prefix}.{name}": param.full_tensor().detach().cpu()
                        if hasattr(param, "full_tensor")
                        else param.detach().cpu()
                        for name, param in sub_lora_params.items()
                    }
                    lora_params.update(sub_lora_params)
                    submodule._is_root = False
                get_torch_device().empty_cache()
    return lora_params


# >>> MARL(adapter_name): extra argument
def collect_lora_params(
    module: FSDP, layered_summon: bool, base_sync_done: bool, adapter_name: str = "default"
) -> OrderedDict:
    # <<< MARL(adapter_name)
    """
    collect lora params or full params if base model is not ready in vllm
    work with if isinstance(self.module._fsdp_wrapped_module, PeftModel)
    """
    from peft.utils.save_and_load import get_peft_model_state_dict

    lora_params = OrderedDict()
    peft_model = getattr(module, "_fsdp_wrapped_module", module)
    if fsdp_version(module) > 0:
        if layered_summon:
            if not base_sync_done:
                raise ValueError(
                    "To use layered_summon, you must make sure base-model is preloaded in vllm, e.g. let "
                    "rollout.load_format=safetensors"
                )
            # >>> MARL(adapter_name)
            lora_params = layered_summon_lora_params(module, adapter_name=adapter_name)
            # <<< MARL(adapter_name)
        else:
            with FSDP.summon_full_params(module, writeback=False):
                if base_sync_done:
                    # >>> MARL(adapter_name)
                    lora_params = get_peft_model_state_dict(peft_model, adapter_name=adapter_name)
                    # <<< MARL(adapter_name)
                    lora_params = {
                        name: param.full_tensor().detach().cpu()
                        if hasattr(param, "full_tensor")
                        else param.detach().cpu()
                        for name, param in lora_params.items()
                    }
                else:
                    model = peft_model.base_model.model
                    orig_dev = "cpu" if "cpu" in str(next(model.parameters()).device) else get_device_name()
                    model = model.to("cpu")
                    for name, param in model.state_dict().items():
                        if any(x in name for x in ["_flat_param", "lora_"]):
                            continue
                        name = name.replace("_fsdp_wrapped_module.", "").replace(".base_layer", "")
                        lora_params[name] = (
                            param.full_tensor().detach().cpu()
                            if hasattr(param, "full_tensor")
                            else param.detach().cpu()
                        )
                    model = model.to(orig_dev)
            get_torch_device().empty_cache()
    else:
        if base_sync_done:
            # >>> MARL(adapter_name)
            lora_params = get_peft_model_state_dict(peft_model, adapter_name=adapter_name)
            # <<< MARL(adapter_name)
        else:
            model = peft_model.base_model.model
            orig_dev = "cpu" if "cpu" in str(next(model.parameters()).device) else get_device_name()
            model = model.to("cpu")
            for name, param in model.state_dict().items():
                if any(x in name for x in ["_flat_param", "lora_"]):
                    continue
                name = name.replace("_fsdp_wrapped_module.", "").replace(".base_layer", "")
                lora_params[name] = param.detach().cpu()
            model = model.to(orig_dev)
    return lora_params
