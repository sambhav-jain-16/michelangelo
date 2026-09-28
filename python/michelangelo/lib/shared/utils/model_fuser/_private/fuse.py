"""Private helpers for :mod:`michelangelo.lib.shared.utils.model_fuser.fuse`.

Module loading/forward-signature helpers that back the public fusing API.
ONNX export helpers have moved to
:mod:`michelangelo.lib.model_manager._private.utils.onnx_utils`, shared with
the non-fused Triton packager. Nothing here is part of the public interface
— import from ``fuse`` instead.
"""

from __future__ import annotations

import inspect
import os
from typing import Any

import torch

from michelangelo.lib.model_manager.schema import DataType, ModelSchema
from michelangelo.lib.model_manager.utils.torch.data_type import (
    data_type_to_torch_dtype,
)
from michelangelo.uniflow.core.utils import import_attribute

from ..fuse_schema import fuse_input_schema
from ..fused_model import FusedModel

# ---------------------------------------------------------------------------
# Module loading / forward-signature helpers
# ---------------------------------------------------------------------------


def _forward_accepts_dict(module: torch.nn.Module) -> bool:
    """Return whether ``module.forward``'s first parameter is dict-annotated."""
    try:
        sig = inspect.signature(module.forward)
    except (ValueError, TypeError):
        return False
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.annotation is inspect.Parameter.empty:
            return False
        return "dict" in str(param.annotation).lower()
    return False


def _forward_param_order(module: torch.nn.Module) -> list[str]:
    """Return ``module.forward``'s parameter names (excluding ``self``), in order."""
    try:
        sig = inspect.signature(module.forward)
    except (ValueError, TypeError):
        return []
    return [name for name in sig.parameters if name != "self"]


def _schema_input_keys(schema: ModelSchema | None) -> list[str]:
    """Return input feature names in schema order, or ``[]`` if unset."""
    if schema is None:
        return []
    return [item.name for item in schema.input_schema]


def _schema_output_keys(schema: ModelSchema | None) -> list[str]:
    """Return output feature names in schema order, or ``[]`` if unset."""
    if schema is None:
        return []
    return [item.name for item in schema.output_schema]


def _schema_input_shapes(schema: ModelSchema | None) -> dict[str, list[int]]:
    """Return each input feature's declared shape (excluding batch), by name.

    A feature whose ``shape`` is ``None`` (genuinely unspecified) is omitted
    -- callers (``FusedModel._reshape_for_predictor``) should leave such a
    feature unreshaped rather than guess. This deliberately differs from
    ``_build_fused_sample_input``, which defaults an unspecified shape to a
    single dimension of size 1 instead of omitting it -- that function must
    still produce a concrete zero-filled tensor for tracing, while reshaping
    an already-real value based on a guessed shape could silently corrupt
    data the schema never actually described. Keep both in sync if either's
    ``None``-handling changes.

    Args:
        schema: A model's input/output schema, or ``None``.

    Returns:
        Mapping of feature name to its declared shape. Empty if ``schema``
        is ``None``.
    """
    if schema is None:
        return {}
    return {
        item.name: list(item.shape)
        for item in schema.input_schema
        if item.shape is not None
    }


def _build_fused_sample_input(
    tx_model_schema: ModelSchema | None,
    model_schema: ModelSchema | None,
    batch_size: int = 1,
) -> dict[str, torch.Tensor]:
    """Build a sample input dict for the fused model, for tracing/inference.

    Uses the same input feature set as :func:`fuse_input_schema`. Each tensor
    has shape ``[batch_size, *feature_shape]`` and dtype derived from the
    item's ``data_type``. Tensors are placed on CUDA when available, else CPU
    (matching the device the fused module is traced on).

    Args:
        tx_model_schema: Native-transform model schema, or ``None``.
        model_schema: Predictor model schema, or ``None``.
        batch_size: Batch dimension for the sample tensors.

    Returns:
        Mapping of fused input feature name to a zero-filled sample tensor.
        Empty if the fused input schema is empty.

    Note:
        Defaults an unspecified (``None``) shape to a single dimension of
        size 1, unlike ``_schema_input_shapes`` (used by
        ``FusedModel._reshape_for_predictor``), which omits such a feature
        instead -- see that function's docstring for why the two
        deliberately disagree. Keep both in sync if either's ``None``
        handling changes.
    """
    input_items = fuse_input_schema(tx_model_schema, model_schema)
    if not input_items:
        return {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample: dict[str, torch.Tensor] = {}
    for item in input_items:
        # `shape=[]` (as opposed to `shape=None`) is a deliberate, documented
        # convention for a true scalar column with no feature dimension at
        # all (see `ColumnConfig.shape`'s docstring: "the common tabular
        # case") -- it must produce a `[batch_size]` sample tensor, not
        # `[batch_size, 1]`. Only a genuinely unset (`None`) shape falls back
        # to a single dimension of size 1.
        feature_shape = [1] if item.shape is None else list(item.shape)
        data_type = item.data_type if item.data_type is not None else DataType.UNKNOWN
        shape = [batch_size] + [max(1, int(s)) for s in feature_shape]
        dtype = data_type_to_torch_dtype(data_type)
        sample[item.name] = torch.zeros(shape, dtype=dtype, device=device)
    return sample


def _is_state_dict(obj: Any) -> bool:
    """Return whether ``obj`` is a state_dict (a dict of name -> Tensor)."""
    return isinstance(obj, dict) and all(
        isinstance(v, torch.Tensor) for v in obj.values()
    )


def _load_module_from_path(
    path: str,
    model_class: str,
    hyperparameters: dict[str, Any],
) -> torch.nn.Module:
    """Load an ``nn.Module`` from a local file (state_dict or full module).

    Args:
        path: Local path to a ``.pt``/``.pth`` file.
        model_class: Dotted class name to instantiate when the file contains
            a state_dict.
        hyperparameters: Constructor kwargs for ``model_class``.

    Returns:
        The loaded module in eval mode.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        TypeError: If the file contains neither a state_dict nor an
            ``nn.Module``.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    # weights_only=False: the file may hold a full nn.Module (not just a
    # state_dict), which torch's restricted unpickler can't reconstruct.
    # Safe only because model artifacts here come from a trusted pipeline
    # (this package's own storage backend), never directly from an
    # unauthenticated end user.
    loaded = torch.load(path, map_location="cpu", weights_only=False)

    if _is_state_dict(loaded):
        model_cls = import_attribute(model_class)
        model = model_cls(**(hyperparameters or {}))
        model.load_state_dict(loaded)
    else:
        model = loaded

    if isinstance(model, torch.nn.Module):
        model.eval()
        return model
    raise TypeError(f"File {path} did not contain a state_dict or nn.Module")


def _align_predictor_input_keys(
    pred_module: torch.nn.Module,
    predictor_input_keys: list[str],
    predictor_takes_dict: bool,
) -> list[str]:
    """Reorder predictor input keys to match ``forward()``'s parameter order.

    For dict-accepting predictors, order does not matter and the keys are
    returned unchanged. For positional predictors, schema keys are aligned to
    the ``forward()`` signature order.

    Args:
        pred_module: The predictor module.
        predictor_input_keys: Feature names from the predictor's input schema.
        predictor_takes_dict: Whether the predictor's ``forward`` takes a
            single dict argument.

    Returns:
        ``predictor_input_keys``, reordered to match ``forward()`` when the
        predictor takes positional tensors.

    Raises:
        ValueError: If a positional predictor's ``forward()`` is missing a
            parameter named in the schema.
    """
    if predictor_takes_dict:
        return predictor_input_keys
    forward_params = _forward_param_order(pred_module)
    schema_set = set(predictor_input_keys)
    if forward_params and schema_set:
        forward_param_set = set(forward_params)
        if not schema_set.issubset(forward_param_set):
            unknown = sorted(schema_set - forward_param_set)
            raise ValueError(
                "Predictor model_schema includes input names that are not "
                f"parameters of forward(): {unknown}. forward() parameters "
                f"(excluding self): {forward_params}. Align the model_schema "
                "with the module's forward(), or use "
                "forward(inputs: dict[str, torch.Tensor]) so the fused model "
                "passes a dict."
            )
        return [p for p in forward_params if p in schema_set]
    return predictor_input_keys


def _build_fused_model_and_sample(
    torch_model_path: str,
    tx_model_path: str,
    model_class: str,
    hyperparameters: dict[str, Any],
    tx_model_class: str,
    tx_hyperparameters: dict[str, Any],
    tx_model_schema: ModelSchema | None = None,
    model_schema: ModelSchema | None = None,
) -> tuple[FusedModel, dict[str, torch.Tensor], list[str]]:
    """Load transform + predictor, build the ``FusedModel``, and a sample batch.

    Args:
        torch_model_path: Local path to the predictor model.
        tx_model_path: Local path to the native-transform model.
        model_class: Dotted class name for the predictor.
        hyperparameters: Constructor kwargs for the predictor.
        tx_model_class: Dotted class name for the transform.
        tx_hyperparameters: Constructor kwargs for the transform.
        tx_model_schema: Native-transform model schema.
        model_schema: Predictor model schema.

    Returns:
        A tuple of ``(fused_module, sample_input, input_key_order)`` where
        ``fused_module`` is on the trace device, ``sample_input`` is a dict
        of sample tensors, and ``input_key_order`` is the sample's key order
        (matching ONNX/Triton input names).

    Raises:
        ValueError: If the fused input schema is empty, so no sample input
            can be built for tracing.
    """
    tx_hyperparameters = tx_hyperparameters or {}
    hyperparameters = hyperparameters or {}

    transform_module = _load_module_from_path(
        tx_model_path, tx_model_class, tx_hyperparameters
    )
    predictor_module = _load_module_from_path(
        torch_model_path, model_class, hyperparameters
    )

    transform_input_keys = _schema_input_keys(tx_model_schema)
    predictor_takes_dict = _forward_accepts_dict(predictor_module)
    predictor_input_keys = _schema_input_keys(model_schema)
    predictor_input_keys = _align_predictor_input_keys(
        predictor_module, predictor_input_keys, predictor_takes_dict
    )

    fused = FusedModel(
        transform_module=transform_module,
        predictor_module=predictor_module,
        transform_input_keys=transform_input_keys,
        predictor_input_keys=predictor_input_keys,
        predictor_takes_dict=predictor_takes_dict,
        predictor_input_shapes=_schema_input_shapes(model_schema),
    )
    fused.eval()

    sample_input = _build_fused_sample_input(tx_model_schema, model_schema)
    if not sample_input:
        raise ValueError(
            "Cannot build sample input for trace: the fused input schema "
            "(from fuse_input_schema) is empty."
        )
    trace_device = next(iter(sample_input.values())).device
    fused = fused.to(trace_device)

    input_key_order = list(sample_input.keys())
    return fused, sample_input, input_key_order


_TORCH_TRANSFORM_MODULE_CLASS = (
    "michelangelo.lib.native_transform.torch.base_transform_module.TorchTransformModule"
)
_LOAD_TRANSFORM_MODULE_FACTORY = (
    "michelangelo.lib.native_transform.torch.base_transform_module."
    "load_transform_module_from_spec_dict"
)


def _is_torch_transform_module(tx_model_class: str) -> bool:
    """Return whether ``tx_model_class`` is ``TorchTransformModule`` or a subclass.

    Checked via ``issubclass`` on the resolved class, not string equality
    against ``_TORCH_TRANSFORM_MODULE_CLASS`` alone, so a downstream
    subclass of ``TorchTransformModule`` also dispatches to
    ``load_transform_module_from_spec_dict`` instead of silently falling
    through to the generic branch (which would build a reconstruction spec
    from its constructor kwargs -- wrong for any ``TorchTransformModule``
    descendant, whose serialized hyperparameters are spec-DAG shaped).
    Resolution happens lazily here (not a module-level import) so
    ``model_fuser`` stays decoupled from ``native_transform`` for callers
    that never fuse a native-transform model.

    Args:
        tx_model_class: Dotted class name of the fused transform model.

    Returns:
        ``True`` if ``tx_model_class`` names ``TorchTransformModule`` or a
        subclass; ``False`` for anything else, including a class that fails
        to import.
    """
    if tx_model_class == _TORCH_TRANSFORM_MODULE_CLASS:
        return True
    try:
        resolved = import_attribute(tx_model_class)
        base = import_attribute(_TORCH_TRANSFORM_MODULE_CLASS)
    except (ImportError, AttributeError, ValueError):
        return False
    return isinstance(resolved, type) and issubclass(resolved, base)


def _build_tx_hydra_spec(
    tx_model_class: str, tx_hyperparameters: dict[str, Any]
) -> dict[str, Any]:
    """Build a Hydra reconstruction spec for a fused native-transform layer stack.

    For the real native-transform case (``tx_model_class`` is
    ``TorchTransformModule`` or a subclass), ``tx_hyperparameters`` is a
    ``TransformSpec.to_dict()``-shaped dict, not constructor kwargs, so it
    can't be passed through as ``{"_target_": tx_model_class,
    **tx_hyperparameters}`` directly. Dispatches instead to
    ``load_transform_module_from_spec_dict``, a factory function that
    rebuilds the ``TransformSpec`` and materializes it. Hydra's generic
    ``instantiate`` resolves ``_target_`` against any importable callable,
    not just a class constructor, so this works the same way a class
    ``_target_`` does.

    For any other ``tx_model_class`` (e.g. a hand-written custom transform
    whose constructor kwargs match its own serialized hyperparameters),
    falls back to the generic ``{"_target_": tx_model_class,
    **tx_hyperparameters}`` spec -- the same shape used for
    ``predictor_module``'s reconstruction spec. This two-way dispatch is a
    deliberate, minimal starting point (one hardcoded special case, checked
    structurally) rather than a general registry: introduce a registry only
    if a second class needs its own bespoke reconstruction.

    Args:
        tx_model_class: Dotted class name of the fused transform model
            (``native_transform_model.metadata.model_class``).
        tx_hyperparameters: The transform model's serialized hyperparameters.

    Returns:
        A Hydra-style reconstruction spec dict with a ``_target_`` key.
    """
    if _is_torch_transform_module(tx_model_class):
        return {
            "_target_": _LOAD_TRANSFORM_MODULE_FACTORY,
            "spec_dict": tx_hyperparameters,
        }
    return {"_target_": tx_model_class, **tx_hyperparameters}
