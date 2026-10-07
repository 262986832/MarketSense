"""决策模型（utility-trainer 任务 T4；与 NanoJev DecisionModel **架构同构**）。

同构来源（只读参考，不 import NanoJev 代码）：
``NanoJev/scripts/train_toy_decisions.py`` L86-141。属性名 / 形状 / 初始化 /
前向逐项对齐，保证 ``state_dict`` 键完全兼容（``backbone.*`` / ``norm.*`` /
``scalar.*`` / ``set_project.*`` / ``set_attention.*`` / ``set_output.*``），
checkpoint 可被 NanoJev 推理脚本只读加载（tech-design D3）。

关键结构（与参考实现逐项一致）：

* ``norm`` = LayerNorm(hidden)；``scalar`` = Linear(hidden, 1)（正态 std=0.02 /
  零偏置初始化——非零随机初始化避免首步死梯度）；
* ``set_head == "attention"``：``set_project`` = Linear(hidden+1, 128)、
  ``set_attention`` = MultiheadAttention(128, 4, dropout=0.0, batch_first=True)、
  ``set_output`` = Linear(128, 1) **零初始化**（只有最终残差投影从零开始，
  上游层非零）；前向 = 最后 token 隐状态 → 候选矩阵 → ``log(k)`` 拼接 →
  集合注意力 → 残差标量 ``index_add``；
* padding 候选以 ``-1e9`` 掩码（softmax 之外）；boolean 题 = 单语义路径单标量，
  表达 logits ``[0, z]``。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DecisionModel", "build_tiny_backbone", "from_pretrained_kwargs"]


class DecisionModel(nn.Module):
    """与 NanoJev DecisionModel 同构的决策头模型（state_dict 键兼容）。"""

    def __init__(self, backbone: nn.Module, set_head: str = "attention") -> None:
        super().__init__()
        if set_head not in ("none", "attention"):
            raise ValueError(f"set_head 必须为 'none' 或 'attention': {set_head!r}")
        self.backbone = backbone
        hidden = backbone.config.hidden_size
        self.norm = nn.LayerNorm(hidden)
        self.scalar = nn.Linear(hidden, 1)  # 非零随机初始化避免首步死梯度
        nn.init.normal_(self.scalar.weight, std=0.02)
        nn.init.zeros_(self.scalar.bias)
        self.set_head = set_head
        if set_head == "attention":
            self.set_project = nn.Linear(hidden + 1, 128)
            self.set_attention = nn.MultiheadAttention(128, 4, dropout=0.0, batch_first=True)
            self.set_output = nn.Linear(128, 1)
            # 只有最终残差投影从零开始；其上游层非零
            nn.init.zeros_(self.set_output.weight)
            nn.init.zeros_(self.set_output.bias)

    def forward(self, examples, pad_token: int):
        paths = [ids for ex in examples for ids in ex["leaf_tokens"]]
        device = self.scalar.weight.device
        lengths = torch.tensor([len(ids) for ids in paths], device=device)
        width = int(lengths.max())
        tokens = torch.full((len(paths), width), pad_token, dtype=torch.long, device=device)
        for i, ids in enumerate(paths):
            tokens[i, : len(ids)] = torch.tensor(ids, device=device)
        attention = torch.arange(width, device=device)[None, :] < lengths[:, None]
        hidden = self.backbone(input_ids=tokens, attention_mask=attention, use_cache=False).last_hidden_state
        leaves = hidden[torch.arange(len(paths), device=device), lengths - 1]
        kmax = max(len(ex["candidate_ids"]) for ex in examples)
        h = leaves.new_zeros((len(examples), kmax, leaves.shape[-1]))
        valid = torch.zeros((len(examples), kmax), dtype=torch.bool, device=device)
        offset = 0
        for i, ex in enumerate(examples):
            n = len(ex["leaf_tokens"])
            h[i, :n] = leaves[offset : offset + n]
            valid[i, : len(ex["candidate_ids"])] = True
            offset += n
        h = self.norm(h)
        z = self.scalar(h).squeeze(-1).float()
        choice = torch.tensor(
            [i for i, ex in enumerate(examples) if ex["type"] == "choice"], device=device
        )
        if self.set_head == "attention" and len(choice):
            log_k = valid[choice].sum(-1).float().log()[:, None, None].expand(-1, kmax, 1)
            u = self.set_project(torch.cat([h[choice], log_k.to(h.dtype)], dim=-1))
            mixed, _ = self.set_attention(u, u, u, key_padding_mask=~valid[choice], need_weights=False)
            delta = self.set_output(torch.tanh(u + mixed)).squeeze(-1).float()
            z = z.index_add(0, choice, delta)
        # Boolean 只有单语义路径与单标量，表达 logits [0, z]
        out = []
        for i, ex in enumerate(examples):
            if ex["type"] == "boolean":
                out.append(F.pad(torch.stack([z[i, 0] * 0, z[i, 0]]), (0, kmax - 2)))
            else:
                out.append(z[i])
        return torch.stack(out).masked_fill(~valid, -1e9), valid


def build_tiny_backbone(
    *,
    vocab_size: int = 1000,
    hidden_size: int = 64,
    num_hidden_layers: int = 2,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 2,
    intermediate_size: int = 128,
    max_position_embeddings: int = 512,
) -> nn.Module:
    """tiny 随机 Qwen3 backbone（``--self-check`` 与单元测试用，不下载任何模型）。

    与预训练 backbone 同构（Qwen3 架构 + sdpa attention），仅规模缩小；
    ``AutoModel.from_config`` 构造，与 NanoJev 加载路径的模块类型一致。
    """
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.for_model(
        "qwen3",
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings,
    )
    config.use_cache = False
    return AutoModel.from_config(config, attn_implementation="sdpa")


def from_pretrained_kwargs(dtype: torch.dtype = torch.float32) -> dict:
    """transformers 版本兼容的预训练加载 kwargs（README §7.4 已记录的坑：
    transformers ≥ 5 用 ``dtype=``，4.51 需要 ``torch_dtype=``）。"""
    import transformers

    major = int(transformers.__version__.split(".", 1)[0])
    key = "dtype" if major >= 5 else "torch_dtype"
    return {key: dtype}
