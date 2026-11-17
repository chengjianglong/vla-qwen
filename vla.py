"""Qwen3-VL based VLA (OpenVLA-OFT style):

  [scene img][wrist img] "Task: ... " <state> <act_1> ... <act_H>
                         Qwen3-VL-2B (LoRA)
  hidden(act_i) -> MLP -> action_t+i   (7-D, normalized), L1 loss

Placeholder tokens (<|quad_start|> = action query, <|quad_end|> = state) are
swapped for learned embeddings by a forward hook on the input embedding layer,
so Qwen3-VL's own image-token handling / M-RoPE is untouched.
"""
import json

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import LoraConfig, get_peft_model

ACT_TOKEN = "<|quad_start|>"
STATE_TOKEN = "<|quad_end|>"


class Normalizer:
    """Map raw values to [-1, 1] via q01/q99 (gripper dims already in [-1, 1])."""

    def __init__(self, stats, skip_last=False):
        self.lo = np.asarray(stats["q01"], np.float32)
        self.hi = np.asarray(stats["q99"], np.float32)
        if skip_last:
            self.lo[-1], self.hi[-1] = -1.0, 1.0

    def norm(self, x):
        return np.clip(2 * (x - self.lo) / (self.hi - self.lo + 1e-8) - 1, -1.5, 1.5)

    def unnorm(self, y):
        return (np.asarray(y) + 1) / 2 * (self.hi - self.lo + 1e-8) + self.lo


class CPUEmbedding(nn.Module):
    """Frozen embedding table kept in CPU RAM (saves ~620MB of VRAM for Qwen3-VL-2B).
    The weight is a plain attribute, not a Parameter, so model.to(cuda) leaves it alone."""

    def __init__(self, emb: nn.Embedding):
        super().__init__()
        self.cpu_weight = emb.weight.detach().cpu()
        self.anchor = nn.Parameter(torch.empty(0, dtype=emb.weight.dtype), requires_grad=False)  # tracks device

    def forward(self, ids):
        out = nn.functional.embedding(ids.cpu(), self.cpu_weight)
        return out.to(self.anchor.device, non_blocking=True)


class QwenVLA(nn.Module):
    def __init__(self, base="Qwen/Qwen3-VL-2B-Instruct", chunk=8, action_dim=7, state_dim=8,
                 lora_r=32, grad_ckpt=True):
        super().__init__()
        self.chunk, self.action_dim = chunk, action_dim
        self.processor = AutoProcessor.from_pretrained(base)
        tok = self.processor.tokenizer
        self.act_id = tok.convert_tokens_to_ids(ACT_TOKEN)
        self.state_id = tok.convert_tokens_to_ids(STATE_TOKEN)
        assert self.act_id != tok.unk_token_id and self.state_id != tok.unk_token_id

        vlm = Qwen3VLForConditionalGeneration.from_pretrained(
            base, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
        vlm.lm_head = nn.Identity()  # we never need vocab logits (saves ~600MB of activations)
        vlm.model.language_model.embed_tokens = CPUEmbedding(vlm.model.language_model.embed_tokens)
        H = vlm.config.text_config.hidden_size
        for p in vlm.parameters():
            p.requires_grad_(False)
        if grad_ckpt:
            vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            vlm.enable_input_require_grads()
        lcfg = LoraConfig(r=lora_r, lora_alpha=lora_r, lora_dropout=0.0,
                          target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
        self.vlm = get_peft_model(vlm, lcfg)

        self.act_queries = nn.Parameter(torch.randn(chunk, H) * 0.02)
        self.state_proj = nn.Sequential(nn.Linear(state_dim, H), nn.GELU(), nn.Linear(H, H))
        self.head = nn.Sequential(nn.LayerNorm(H), nn.Linear(H, 1024), nn.GELU(), nn.Linear(1024, action_dim))

        self._state = None
        self.vlm.get_input_embeddings().register_forward_hook(self._embed_hook)

    # -- swap placeholder embeddings for learned ones -------------------------------
    def _embed_hook(self, module, inputs, out):
        if self._state is None:
            return out
        ids = inputs[0]
        out = out.clone()
        act_mask = ids == self.act_id
        B = ids.shape[0]
        out[act_mask] = self.act_queries.to(out.dtype).repeat(B, 1)
        out[ids == self.state_id] = self.state_proj(self._state.float()).to(out.dtype)
        return out

    def build_prompt(self, task):
        text = (f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                f"<|vision_start|><|image_pad|><|vision_end|>"
                f"Task: {task.strip().lower()}\nState: {STATE_TOKEN}\nActions: "
                + ACT_TOKEN * self.chunk)
        return text

    def encode(self, images, tasks):
        """images: list of [scene, wrist] PIL/np uint8 HxWx3; tasks: list[str]."""
        flat = [im for pair in images for im in pair]
        texts = [self.build_prompt(t) for t in tasks]
        # left padding so the action tokens are always the last `chunk` positions
        self.processor.tokenizer.padding_side = "left"
        return self.processor(text=texts, images=flat, return_tensors="pt", padding=True)

    def forward(self, batch, state):
        self._state = state
        try:
            out = self.vlm(**batch, output_hidden_states=False, use_cache=False)
        finally:
            self._state = None
        h = out.logits  # lm_head is Identity -> last hidden states [B, T, H]
        h = h[:, -self.chunk:]
        return self.head(h.float())  # [B, chunk, action_dim]

    def trainable_state_dict(self):
        return {k: v for k, v in self.state_dict().items()
                if "lora_" in k or k.startswith(("act_queries", "state_proj", "head"))}
