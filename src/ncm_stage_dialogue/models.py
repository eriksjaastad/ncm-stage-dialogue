"""MLX implementations of TextNCA and its autoregressive baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten

from ncm_stage_dialogue.experiment_data import VOCAB_SIZE


class ModelConfigError(ValueError):
    """Raised when an experiment model configuration is invalid."""


@dataclass(frozen=True)
class ModelConfig:
    """Architecture parameters captured with every experiment run."""

    model_type: str
    context_length: int
    hidden_size: int
    num_heads: int
    ffn_size: int
    vocab_size: int = VOCAB_SIZE
    windows: tuple[int, ...] = (4, 16, 64)
    iterations_per_stage: int = 4
    num_layers: int = 5

    def validate(self) -> None:
        if self.model_type not in {"textnca", "transformer"}:
            raise ModelConfigError(f"unknown model_type: {self.model_type}")
        for name in ("context_length", "hidden_size", "num_heads", "ffn_size", "vocab_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ModelConfigError(f"{name} must be a positive integer")
        if self.hidden_size % self.num_heads:
            raise ModelConfigError("hidden_size must be divisible by num_heads")
        if self.model_type == "textnca":
            if not self.windows or any(
                type(window) is not int or window < 1 for window in self.windows
            ):
                raise ModelConfigError("TextNCA windows must be positive integers")
            if type(self.iterations_per_stage) is not int or self.iterations_per_stage < 1:
                raise ModelConfigError("iterations_per_stage must be positive")
        if self.model_type == "transformer" and (
            type(self.num_layers) is not int or self.num_layers < 1
        ):
            raise ModelConfigError("num_layers must be positive")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["windows"] = list(self.windows)
        return result

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ModelConfig:
        value = dict(raw)
        if "windows" in value:
            value["windows"] = tuple(value["windows"])
        config = cls(**value)
        config.validate()
        return config


def causal_local_mask(length: int, window: int) -> mx.array:
    """Return an additive mask allowing only the current and prior window tokens."""
    if length < 1 or window < 1:
        raise ModelConfigError("mask length and window must be positive")
    positions = mx.arange(length)
    distance = positions[:, None] - positions[None, :]
    allowed = (distance >= 0) & (distance < window)
    return mx.where(allowed, mx.array(0.0), mx.array(float("-inf")))


class FeedForward(nn.Module):
    """Two-layer SiLU feed-forward network."""

    def __init__(self, hidden_size: int, ffn_size: int) -> None:
        super().__init__()
        self.up = nn.Linear(hidden_size, ffn_size)
        self.down = nn.Linear(ffn_size, hidden_size)

    def __call__(self, inputs: mx.array) -> mx.array:
        return self.down(nn.silu(self.up(inputs)))


class CellGRU(nn.Module):
    """GRU-style per-token state update without sequence-axis recurrence."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.update_gate = nn.Linear(2 * hidden_size, hidden_size)
        self.reset_gate = nn.Linear(2 * hidden_size, hidden_size)
        self.candidate = nn.Linear(2 * hidden_size, hidden_size)

    def __call__(self, state: mx.array, update: mx.array) -> mx.array:
        combined = mx.concatenate((state, update), axis=-1)
        update_weight = mx.sigmoid(self.update_gate(combined))
        reset_weight = mx.sigmoid(self.reset_gate(combined))
        candidate_input = mx.concatenate((reset_weight * state, update), axis=-1)
        candidate = mx.tanh(self.candidate(candidate_input))
        return (1.0 - update_weight) * state + update_weight * candidate


class TextNCAStage(nn.Module):
    """One local-perception, feed-forward, and gated cellular update stage."""

    def __init__(self, hidden_size: int, num_heads: int, ffn_size: int, window: int) -> None:
        super().__init__()
        self.window = window
        self.attention = nn.MultiHeadAttention(hidden_size, num_heads, bias=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.feed_forward = FeedForward(hidden_size, ffn_size)
        self.gate = CellGRU(hidden_size)

    def __call__(self, state: mx.array, step_embedding: mx.array) -> mx.array:
        perceived_input = state + step_embedding
        mask = causal_local_mask(state.shape[1], self.window)
        perceived = self.attention(perceived_input, perceived_input, perceived_input, mask=mask)
        update = self.feed_forward(self.norm(perceived))
        return self.gate(state, update)


class TextNCA(nn.Module):
    """Hierarchical local-attention neural cellular language model."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        if config.model_type != "textnca":
            raise ModelConfigError("TextNCA requires model_type=textnca")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.context_length, config.hidden_size)
        self.stages = [
            TextNCAStage(config.hidden_size, config.num_heads, config.ffn_size, window)
            for window in config.windows
        ]
        step_count = len(config.windows) * config.iterations_per_stage
        scale = config.hidden_size**-0.5
        self.step_embeddings = mx.random.normal((step_count, config.hidden_size)) * scale
        self.output_norm = nn.LayerNorm(config.hidden_size)

    def __call__(self, tokens: mx.array) -> mx.array:
        if tokens.ndim != 2 or tokens.shape[1] > self.config.context_length:
            raise ValueError("tokens must have shape [batch, length] within context_length")
        positions = mx.arange(tokens.shape[1])
        state = self.token_embedding(tokens) + self.position_embedding(positions)
        step_index = 0
        for stage in self.stages:
            for _ in range(self.config.iterations_per_stage):
                state = stage(state, self.step_embeddings[step_index])
                step_index += 1
        return self.token_embedding.as_linear(self.output_norm(state))


class TransformerBlock(nn.Module):
    """Pre-norm global causal Transformer block."""

    def __init__(self, hidden_size: int, num_heads: int, ffn_size: int) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.attention = nn.MultiHeadAttention(hidden_size, num_heads, bias=True)
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.feed_forward = FeedForward(hidden_size, ffn_size)

    def __call__(self, state: mx.array, mask: mx.array) -> mx.array:
        normalized = self.attention_norm(state)
        state = state + self.attention(normalized, normalized, normalized, mask=mask)
        return state + self.feed_forward(self.ffn_norm(state))


class TransformerLM(nn.Module):
    """Parameter-matched autoregressive Transformer baseline."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        if config.model_type != "transformer":
            raise ModelConfigError("TransformerLM requires model_type=transformer")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.context_length, config.hidden_size)
        self.layers = [
            TransformerBlock(config.hidden_size, config.num_heads, config.ffn_size)
            for _ in range(config.num_layers)
        ]
        self.output_norm = nn.LayerNorm(config.hidden_size)

    def __call__(self, tokens: mx.array) -> mx.array:
        if tokens.ndim != 2 or tokens.shape[1] > self.config.context_length:
            raise ValueError("tokens must have shape [batch, length] within context_length")
        positions = mx.arange(tokens.shape[1])
        state = self.token_embedding(tokens) + self.position_embedding(positions)
        mask = nn.MultiHeadAttention.create_additive_causal_mask(tokens.shape[1])
        for layer in self.layers:
            state = layer(state, mask)
        return self.token_embedding.as_linear(self.output_norm(state))


def build_model(config: ModelConfig) -> TextNCA | TransformerLM:
    """Construct the declared model architecture."""
    config.validate()
    if config.model_type == "textnca":
        return TextNCA(config)
    return TransformerLM(config)


def parameter_count(model: nn.Module) -> int:
    """Count scalar trainable parameters in an MLX module tree."""
    return sum(array.size for _, array in tree_flatten(model.trainable_parameters()))
