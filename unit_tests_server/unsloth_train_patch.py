# stolen from ART https://github.com/OpenPipe/ART/blob/5f3dea20069ee8e4afbd482e529df5ee80d81b81/src/art/local/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Callable, Dict, List, Optional, Tuple, Union

from trl import GRPOTrainer
from peft import PeftModel
import numpy as np
import os

import pdb

torch_compile_options = {
    "epilogue_fusion"   : True,
    "max_autotune"      : False,
    "shape_padding"     : True,
    "trace.enabled"     : False,
    "triton.cudagraphs" : False,
}

def train(
    trainer: "GRPOTrainer",
) -> None:
    _compute_loss = trainer.compute_loss
    trainer.compute_loss = lambda *args, **kwargs: compute_loss(trainer, *args, **kwargs)
    # trainer.log = get_log_fn(trainer, results_queue)
    try:
        trainer.train()
    finally:
        trainer.compute_loss = _compute_loss
        # trainer.log = _log

@torch.compile(dynamic = True, fullgraph = True, options = torch_compile_options)
def grpo_compute_loss_slow(old_logits, new_logits, input_ids, mask, beta, advantages):
    # All Unsloth Zoo code licensed under LGPLv3
    input_ids  = input_ids.unsqueeze(-1)
    new_logits = new_logits.to(torch.float32)
    new_x = torch.gather(new_logits, dim = -1, index = input_ids).squeeze(-1)
    new = new_x - torch.logsumexp(new_logits, dim = -1)

    if old_logits is None:
        kl_i = torch.zeros_like(mask)
    else:
        old_logits = old_logits.to(torch.float32)

        # x_i - logsumexp(x_i)
        old_x = torch.gather(old_logits, dim = -1, index = input_ids).squeeze(-1)
        old = old_x - torch.logsumexp(old_logits, dim = -1)

        # Reverse KL
        kl_i = torch.exp(old - new) - (old - new) - 1.0
    # Full correct reverse KL divergence?? Missing term maybe?
    # kl_i = torch.exp(new) * kl_i

    # Below is forward KL (normal KL)
    # kl_i = torch.exp(old) * (old - new)

    # Must detach - otherwise gradients are not propagated correctly!
    # exp(x - x) == 1
    loss_i = torch.exp(new - new.detach()) * advantages.unsqueeze(1)
    loss_i = -(loss_i - beta * kl_i)

    mask = mask.to(torch.float32)
    n_mask_per_reward = mask.sum(1)

    # See https://github.com/huggingface/trl/pull/2881
    loss_per_reward = (loss_i * mask).sum(1) / n_mask_per_reward
    loss = loss_per_reward.mean()
    # loss = (loss_i * mask).sum() / mask.sum()
    
    # Get metrics as well which are folded
    with torch.inference_mode():
        completion_length = n_mask_per_reward.mean()
        mean_kl_per_reward = (kl_i * mask).sum(1) / n_mask_per_reward
        mean_kl = mean_kl_per_reward.mean()

    return loss, completion_length, 
    
def grpo_accumulated_loss(
    trainer,
    input_ids,
    logits_to_keep,
    completion_mask,
    advantages,
    n_chunks = -1,
):
    # All Unsloth Zoo code licensed under LGPLv3
    bsz, qlen = input_ids.shape
    # Find closest multiple
    factors = [i for i in range(1, bsz + 1) if bsz % i == 0]
    if n_chunks == -1: n_chunks = bsz
    n_chunks = factors[min(np.searchsorted(factors, n_chunks), len(factors)-1)]

    mixed_dtype = torch.float16 if os.environ.get('ACCELERATE_MIXED_PRECISION', 'fp16') == 'fp16' else torch.bfloat16
    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"

    completion_input_ids = input_ids[:, -logits_to_keep:]
    lm_head = trainer.model.get_output_embeddings().weight

    with torch.amp.autocast(device_type = "cuda", dtype = mixed_dtype):
        with torch.inference_mode(), trainer.accelerator.unwrap_model(trainer.model, keep_fp32_wrapper = False).disable_adapter():
            old_hidden_states = trainer.model(input_ids = input_ids, logits_to_keep = logits_to_keep + 1).logits
        pass

        new_hidden_states = trainer.model(input_ids = input_ids, logits_to_keep = logits_to_keep + 1).logits
        
        loss, completion_length, mean_kl = UnslothEfficientGRPO.apply(
            new_hidden_states, old_hidden_states, lm_head,
            completion_input_ids, completion_mask, advantages, trainer.beta,
            trainer.accelerator.scaler,
            n_chunks, 
        )
        return loss, completion_length, mean_kl

        # Old non efficient code path
        new_logits = torch.matmul(new_hidden_states, lm_head.t())
        new_logits = new_logits[:, :-1, :] # exclude the last logit: it corresponds to the next token pred
        old_logits = torch.matmul(old_hidden_states, lm_head.t())
        old_logits = old_logits[:, :-1, :] # exclude the last logit: it corresponds to the next token pred
        loss, completion_length, mean_kl = grpo_compute_loss(
            old_logits, new_logits, completion_input_ids, completion_mask, trainer.beta, advantages,
        )
        return loss, completion_length, mean_kl
    pass

def compute_loss(self, model, inputs, return_outputs = False, num_items_in_batch = None):
    if return_outputs:
        raise ValueError("The GRPOTrainer does not support returning outputs")
    # Compute the per-token log probabilities for the model

    prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
    completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    bsz, qlen = input_ids.shape
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    # attention_mask = None
    logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
    _input_ids = input_ids
    _logits_to_keep = logits_to_keep
    
    per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

    # Compute the KL divergence between the model and the reference model
    ref_per_token_logps = inputs["ref_per_token_logps"] if "ref_per_token_logps" in inputs else None
    # per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

    # x - x.detach() allows for preserving gradients from x
    advantages = inputs["advantages"]
    # per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
    # per_token_loss = -(per_token_loss - self.beta * per_token_kl)
    # loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    input_ids = input_ids[:, -logits_to_keep:]
    if per_token_logps is not None:
        loss, completion_length, mean_kl = grpo_compute_loss_slow(
            ref_per_token_logps, per_token_logps, input_ids, completion_mask, self.beta, advantages,
        )
    else:
        loss, completion_length, mean_kl = grpo_accumulated_loss(
            self, _input_ids, logits_to_keep, completion_mask, advantages,
            n_chunks = self.args.unsloth_num_chunks,
        )

    # Log the metrics
    # completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()

    # mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    # self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

    if "train" in self._metrics:
        mode = "eval" if self.control.should_evaluate else "train"
        self._metrics[mode]["completion_length"].append(completion_length.item())
        self._metrics[mode]["kl"].append(mean_kl.item())
    else:
        self._metrics["completion_length"].append(completion_length.item())
        self._metrics["kl"].append(mean_kl.item())
    return loss
