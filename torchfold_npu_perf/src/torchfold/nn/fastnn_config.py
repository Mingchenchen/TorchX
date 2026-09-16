import contextlib
import contextvars


# options: ["torch"]
layer_norm_implementation = "torch"

# options: ["torch", "Fusion_Attention"]
dot_product_attention_implementation = "torch"


_single_card_valid_token_count = contextvars.ContextVar(
    "torchfold_single_card_valid_token_count",
    default=None,
)


@contextlib.contextmanager
def single_card_fa_mask_context(valid_token_count):
    """Expose a host-derived valid-token prefix to single-card FA callers."""
    if valid_token_count is not None:
        valid_token_count = int(valid_token_count)
        if valid_token_count < 0:
            raise ValueError(
                "single-card valid token count must be non-negative, "
                f"got {valid_token_count}"
            )
    token = _single_card_valid_token_count.set(valid_token_count)
    try:
        yield
    finally:
        _single_card_valid_token_count.reset(token)


def single_card_valid_token_count():
    """Return the host-proven valid-token prefix for the active inference."""
    return _single_card_valid_token_count.get()
