"""torchfold.model.build — model factory + registry dispatch.

build_model(model_name, ...) looks up configs_model_type.model_configs[model_name],
imports its `model_class` (an "module:ClassName" path), constructs it from the
runtime kwargs, and loads params per `param_format`. This is the single switch
point for selecting a model (`--model_name`), including ENTIRELY DIFFERENT model
classes (register a new entry in configs/configs_model_type.py).

Default registry key is `torchfold_af3_default`. When `load_params` is true and
`param_format` is a JAX dump, `import_jax_weights_` maps that dump into the
torch module. When `load_params` is false (e.g. a PyTorch checkpoint follows),
the module is built without reading a dump.
"""
import importlib
from typing import Any, Callable, Optional


def _import_obj(path: str):
    """Import an object from a 'module.sub:ClassName' path."""
    if ":" not in path:
        raise ValueError(f"model_class must be 'module:ClassName', got {path!r}")
    module_path, obj_name = path.split(":", 1)
    return getattr(importlib.import_module(module_path), obj_name)


def build_model(
    model_name: str,
    model_configs: dict,
    *,
    runtime_kwargs: dict,
    af3_params_dir: Optional[str] = None,
    load_params: bool = True,
    log: Callable[[str], Any] = print,
):
    """Construct the model selected by `model_name`.

    Args:
        model_name: key into model_configs (from configs/configs_model_type.py).
        model_configs: the registry dict.
        runtime_kwargs: kwargs passed to the model class constructor.
        af3_params_dir: JAX dump directory (for param_format == af3_bin_zst).
        load_params: if False, skip param loading (e.g. for a smoke/build test).
        log: logger callable.
    """
    if model_name not in model_configs:
        raise ValueError(
            f"unknown model_name {model_name!r}. Registered: {list(model_configs)}. "
            f"Register new models in configs/configs_model_type.py."
        )
    spec = model_configs[model_name]
    model_class_path = spec.get("model_class")
    if not model_class_path:
        raise ValueError(f"model_configs[{model_name!r}] has no 'model_class'")
    ModelClass = _import_obj(model_class_path)
    model = ModelClass(**runtime_kwargs)

    fmt = spec.get("param_format")
    if load_params and fmt == "af3_bin_zst":
        if af3_params_dir is None:
            raise ValueError("af3_params_dir required for param_format 'af3_bin_zst'")
        from torchfold.weights.import_jax_weights import import_jax_weights_
        import_jax_weights_(model, af3_params_dir)
        log(f"[build_model] {model_name}: {ModelClass.__name__} + jax dump <- {af3_params_dir}")
    elif load_params and fmt == "torch_state_dict":
        raise NotImplementedError(
            f"param_format 'torch_state_dict' (model {model_name}) not implemented yet"
        )
    elif load_params and fmt:
        raise ValueError(f"unknown param_format {fmt!r} for model {model_name}")
    else:
        log(f"[build_model] {model_name}: {ModelClass.__name__} (no params loaded)")
    return model
