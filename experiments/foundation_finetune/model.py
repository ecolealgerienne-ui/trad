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
        extra_dim: int = 0,
        device: str = "cpu",
    ):
        super().__init__()
        self.model_name = model_name
        self.tokenizer, self.t5, d_model = load_chronos(model_name, device=device)
        self.d_model = d_model
        self.extra_dim = extra_dim

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
            nn.Linear(d_model + extra_dim, head_hidden),
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
            x_rsi: (batch, seq_len) float tensor (CPU or CUDA).
        Returns:
            input_ids       : (batch, seq_len_padded) long
            attention_mask  : (batch, seq_len_padded) bool/long
            tokenizer_state : container with mean/scale used (for unscaling)

        Note: Chronos tokenizer keeps quantile boundaries on CPU, so
        torch.bucketize requires a CPU input. We force CPU here; the
        returned ids/mask are moved back to GPU in forward().
        """
        x_cpu = x_rsi.detach().cpu() if x_rsi.is_cuda else x_rsi
        ids, attn_mask, state = self.tokenizer.context_input_transform(x_cpu)
        return ids, attn_mask, state

    def forward(self, x_rsi: torch.Tensor, extras: torch.Tensor = None) -> torch.Tensor:
        """Forward.

        Args:
            x_rsi : (batch, seq_len) float tensor (RSI window, CPU or CUDA).
            extras: (batch, extra_dim) float tensor or None. Required iff
                    extra_dim > 0. Concatenated to the pooled T5 embedding
                    before the regression head.
        Returns:
            (batch,) predicted slope scalar.
        """
        ids, attn_mask, _ = self.tokenize(x_rsi)
        ids = ids.to(self._device())
        attn_mask = attn_mask.to(self._device())

        out = self.encoder(input_ids=ids, attention_mask=attn_mask, return_dict=True)
        hidden = out.last_hidden_state  # (B, T, d_model)

        mask = attn_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        if self.extra_dim > 0:
            if extras is None:
                raise ValueError(
                    f"This model expects extras of dim {self.extra_dim}, got None."
                )
            extras = extras.to(pooled.device, dtype=pooled.dtype)
            if extras.shape[-1] != self.extra_dim:
                raise ValueError(
                    f"extras dim {extras.shape[-1]} != expected {self.extra_dim}"
                )
            pooled = torch.cat([pooled, extras], dim=-1)
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
    p.add_argument("--extra-dim", type=int, default=0,
                   help="If >0, model expects extras tensor of this dim in forward.")
    args = p.parse_args()

    kwargs = dict(model_name=args.model, head_hidden=64, extra_dim=args.extra_dim)
    if args.mode == "probing":
        kwargs.update(freeze_backbone=True, use_lora=False)
    elif args.mode == "lora":
        kwargs.update(freeze_backbone=True, use_lora=True, lora_rank=args.lora_rank)
    else:
        kwargs.update(freeze_backbone=False, use_lora=False)

    print(f"Loading {args.model} (mode={args.mode}, extra_dim={args.extra_dim})...")
    model = ChronosRegressor(**kwargs)
    print(f"  d_model={model.d_model}")
    print(f"  trainable params: {model.count_trainable():,}")
    print(f"  total params:     {model.count_total():,}")

    x = torch.rand(args.batch, args.window) * 100.0  # mock RSI [0, 100]
    extras = torch.randn(args.batch, args.extra_dim) if args.extra_dim > 0 else None
    print(f"\nForward x: shape={tuple(x.shape)} mean={x.mean():.2f} std={x.std():.2f}"
          + (f"  extras: shape={tuple(extras.shape)}" if extras is not None else ""))
    with torch.no_grad():
        y = model(x, extras) if extras is not None else model(x)
    print(f"Output y: shape={tuple(y.shape)} mean={y.mean():.4f} std={y.std():.4f}")
    print("Smoke test OK.")


if __name__ == "__main__":
    main()
