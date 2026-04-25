"""ChronosRegressor: Chronos T5 encoder + MLP regression head for slope prediction.

Architecture:
    RSI[t-95:t]  (batch, 96) float in [0, 100]
        |
        v  (Chronos quantile tokenizer: mean-scaling + bin)
    input_ids    (batch, 96+1) int  (Chronos prepends EOS)
        |
        v  (T5 encoder, possibly with LoRA on q/v)
    hidden       (batch, seq_len, d_model)
        |
        v  (mean pool over attended positions)
    pooled       (batch, d_model)
        |
        v  (MLP head: Linear -> GELU -> Dropout -> Linear)
    slope_pred   (batch,)

Modes:
    --freeze-backbone : all T5 weights frozen, only MLP head trains (PROBING)
    --use-lora        : LoRA adapters on q/v of T5 attention, MLP head trains
                        (overrides freeze; trainable params = LoRA + head)

Default: probing mode (freeze=True, use_lora=False).

Smoke test:
    python experiments/foundation_finetune/model.py \
        --model amazon/chronos-t5-tiny --batch 4
"""

import argparse

import torch
import torch.nn as nn


def load_chronos(model_name: str, device: str = "cpu", dtype=torch.float32):
    """Load Chronos pipeline and extract (tokenizer, T5 inner model, d_model)."""
    try:
        from chronos import ChronosPipeline
    except ImportError as e:
        raise ImportError(
            "chronos-forecasting not installed. Run: pip install chronos-forecasting"
        ) from e

    pipeline = ChronosPipeline.from_pretrained(
        model_name, device_map=device, torch_dtype=dtype
    )
    tokenizer = pipeline.tokenizer
    # pipeline.model is ChronosModel wrapper; pipeline.model.model is the inner T5
    inner_t5 = pipeline.model.model
    d_model = inner_t5.config.d_model
    return tokenizer, inner_t5, d_model


class ChronosRegressor(nn.Module):
    """Chronos T5 encoder + regression head -> scalar slope prediction."""

    def __init__(
        self,
        model_name: str = "amazon/chronos-t5-tiny",
        freeze_backbone: bool = True,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        head_hidden: int = 64,
        head_dropout: float = 0.1,
        device: str = "cpu",
    ):
        super().__init__()
        self.model_name = model_name
        self.tokenizer, self.t5, d_model = load_chronos(model_name, device=device)
        self.d_model = d_model

        if use_lora:
            from peft import LoraConfig, get_peft_model

            lora_cfg = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=["q", "v"],
                lora_dropout=lora_dropout,
                bias="none",
                task_type="SEQ_2_SEQ_LM",
            )
            self.t5 = get_peft_model(self.t5, lora_cfg)
        elif freeze_backbone:
            for p in self.t5.parameters():
                p.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(d_model, head_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 1),
        )

    @property
    def encoder(self):
        """Return underlying T5 encoder (handles peft wrapping transparently)."""
        if hasattr(self.t5, "get_base_model"):  # PeftModel
            return self.t5.get_base_model().encoder
        return self.t5.encoder

    def tokenize(self, x_rsi: torch.Tensor):
        """Tokenize a batch of RSI windows via Chronos quantile tokenizer.

        Args:
            x_rsi: (batch, seq_len) float tensor.
        Returns:
            input_ids       : (batch, seq_len_padded) long
            attention_mask  : (batch, seq_len_padded) bool/long
            tokenizer_state : container with mean/scale used (for unscaling)
        """
        # ChronosTokenizer.context_input_transform expects a 2D float tensor
        ids, attn_mask, state = self.tokenizer.context_input_transform(x_rsi)
        return ids, attn_mask, state

    def forward(self, x_rsi: torch.Tensor) -> torch.Tensor:
        """x_rsi: (batch, 96) float tensor. Returns: (batch,) predicted slope."""
        ids, attn_mask, _ = self.tokenize(x_rsi)
        ids = ids.to(self._device())
        attn_mask = attn_mask.to(self._device())

        out = self.encoder(input_ids=ids, attention_mask=attn_mask, return_dict=True)
        hidden = out.last_hidden_state  # (B, T, d_model)

        mask = attn_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        return self.head(pooled).squeeze(-1)

    def _device(self):
        return next(self.parameters()).device

    def count_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total(self):
        return sum(p.numel() for p in self.parameters())


def main():
    p = argparse.ArgumentParser(description="Smoke test ChronosRegressor.")
    p.add_argument("--model", default="amazon/chronos-t5-tiny",
                   choices=["amazon/chronos-t5-tiny",
                            "amazon/chronos-t5-mini",
                            "amazon/chronos-t5-small",
                            "amazon/chronos-t5-base"])
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--window", type=int, default=96)
    p.add_argument("--mode", default="probing",
                   choices=["probing", "lora", "full"])
    p.add_argument("--lora-rank", type=int, default=8)
    args = p.parse_args()

    kwargs = dict(model_name=args.model, head_hidden=64)
    if args.mode == "probing":
        kwargs.update(freeze_backbone=True, use_lora=False)
    elif args.mode == "lora":
        kwargs.update(freeze_backbone=True, use_lora=True, lora_rank=args.lora_rank)
    else:
        kwargs.update(freeze_backbone=False, use_lora=False)

    print(f"Loading {args.model} (mode={args.mode})...")
    model = ChronosRegressor(**kwargs)
    print(f"  d_model={model.d_model}")
    print(f"  trainable params: {model.count_trainable():,}")
    print(f"  total params:     {model.count_total():,}")

    x = torch.rand(args.batch, args.window) * 100.0  # mock RSI [0, 100]
    print(f"\nForward x: shape={tuple(x.shape)} mean={x.mean():.2f} std={x.std():.2f}")
    with torch.no_grad():
        y = model(x)
    print(f"Output y: shape={tuple(y.shape)} mean={y.mean():.4f} std={y.std():.4f}")
    print("Smoke test OK.")


if __name__ == "__main__":
    main()
