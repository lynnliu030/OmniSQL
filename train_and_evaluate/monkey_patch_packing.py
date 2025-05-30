'''
This script originates from the GitHub repository:
https://github.com/MeetKai/functionary/blob/main/functionary/train/packing/monkey_patch_packing.py
'''

import torch
import torch.nn.functional as F
import transformers
from typing import Optional
import sys


def get_max_seqlen_in_batch(attention_mask):
    max_num = torch.max(attention_mask)
    # attention_mask: B x N
    counts = []
    for i in range(1, max_num + 1):
        counts.append(
            torch.sum(attention_mask == i, axis=-1)
        )  # shape: B, count length of data point maksed with i
    result = torch.stack(counts, axis=1)
    result = result.flatten()
    return result[result.nonzero()].squeeze(-1).to(dtype=torch.int32)


def get_unpad_data(attention_mask):
    seqlens_in_batch = get_max_seqlen_in_batch(
        attention_mask
    )  # attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(
        torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0)
    )
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


# Copy from original implementation of modeling_mixtral.py from transformers, Just change a little bit with new_attention_mask
def load_balancing_loss_func(
    gate_logits: torch.Tensor,
    num_experts: torch.Tensor = None,
    top_k=2,
    attention_mask: Optional[torch.Tensor] = None,
) -> float:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits (Union[`torch.Tensor`, Tuple[torch.Tensor]):
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        attention_mask (`torch.Tensor`, None):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.
        num_experts (`int`, *optional*):
            Number of experts

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat(
            [layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0
        )

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        # ONLY ADD THIS LINE OF CODE, AND REPLACE attention_mask WITH new_attention_mask
        new_attention_mask = (attention_mask != 0).int().to(attention_mask.device)
        batch_size, sequence_length = new_attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (
            batch_size * sequence_length
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            new_attention_mask[None, :, :, None, None]
            .expand(
                (num_hidden_layers, batch_size, sequence_length, top_k, num_experts)
            )
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(
            expert_mask.float() * expert_attention_mask, dim=0
        ) / torch.sum(expert_attention_mask, dim=0)

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            new_attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(
            routing_weights * router_per_expert_attention_mask, dim=0
        ) / torch.sum(router_per_expert_attention_mask, dim=0)

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


def monkey_patch_for_model_with_name(model_type: str, modelling_type: str):
    """For example for llama: model_package = llama, modelling_module=modeling_llama

    Args:
        model_package (_type_): _description_
        modelling_module (_type_): _description_
    """
    module = getattr(getattr(transformers, model_type), modelling_type)
    if hasattr(module, "_get_unpad_data"):
        module._get_unpad_data = get_unpad_data
    print(
        f"cannot packing llama because _get_unpad_data was not found in transformers.{model_type}.{modelling_type}.py or transformers.modeling_flash_attention_utils._get_unpad_data"
    )
    sys.exit(1)


def monkey_patch_packing_for_model(pretrained_model):

    # Monkey-patch flash attention if this transformers already merged: https://github.com/huggingface/transformers/commit/e314395277d784a34ee99526f48155d4d62cff3d
    # this will work for all models using flash attention: Llama, Mistral, Qwen2, Phi3, ...
    model_config = transformers.AutoConfig.from_pretrained(pretrained_model)
    config_type = type(model_config).__name__.lower()
    if hasattr(transformers, "modeling_flash_attention_utils"):
        transformers.modeling_flash_attention_utils._get_unpad_data = get_unpad_data
    else:  # if this is the old version of transformer
        model_type, modelling_type = "", ""
        if config_type == "mistralconfig":
            print("monkey_patch_packing for Mistral ")
            transformers.models.mistral.modeling_mistral._get_unpad_data = (
                get_unpad_data
            )

        elif config_type == "llamaconfig":
            print("monkey_patch_packing for Llama ")
            transformers.models.llama.modeling_llama._get_unpad_data = get_unpad_data

        elif config_type == "mixtralconfig":
            print("monkey_patch_packing for Mixtral")
            transformers.models.mixtral.modeling_mixtral._get_unpad_data = (
                get_unpad_data
            )

        elif config_type == "qwen2config":
            print("monkey_patch_packing for Qwen2")
            # transformers.models.qwen2.modeling_qwen2
            model_type, modelling_type = "qwen2", "modeling_qwen2"
            transformers.models.qwen2.modeling_qwen2._get_unpad_data = get_unpad_data

        elif config_type == "phi3config":
            # transformers.models.phi3.modeling_phi3
            print("monkey_patch_packing for Qwen2")
            transformers.models.phi3.modeling_phi3._get_unpad_data = get_unpad_data
        else:
            raise Exception(
                f"{config_type} is not supported, currently we only support: Mistral, Mixtral, Llama, Qwen2 for monkey-patch-packing"
            )

        monkey_patch_for_model_with_name(model_type, modelling_type)

    if config_type == "mixtralconfig":
        # if it is mixtral, we need to monkey-patch the load_balancing_loss_func
        transformers.models.mixtral.modeling_mixtral.load_balancing_loss_func = (
            load_balancing_loss_func
        )