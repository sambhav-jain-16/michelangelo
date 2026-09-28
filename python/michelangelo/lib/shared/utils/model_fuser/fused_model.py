"""Fused model composing a native-transform module and a predictor module.

``forward`` accepts ``dict[str, Tensor]`` keyed by feature name. The transform
runs on its input schema; its output is merged with passthrough features from
the input (the predictor receives the transform's output where available,
else the original input for that feature). Output is the predictor's output
tensor. Designed to be TorchScript-exportable.

The transform and predictor are independently-schema'd models -- a value the
transform produces (or passes through) for a given feature name is not
guaranteed to already be shaped the way the predictor's own schema declares
that same feature. For example, ``tabular_native_transform``'s own scalar
convention is a declared shape of ``[1]`` (see its ``_to_batched_tensor``),
while ``tabular_trainer``'s ``ColumnConfig`` documents ``[]`` (no feature
dimension) as its scalar convention -- both valid, independently-documented
conventions, so ``FusedModel`` reshapes each value to the predictor's own
declared per-feature shape (when known) right before handing it off, rather
than assuming the two schemas already agree.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["FusedModel"]


def _is_safe_reshape(source_shape: list[int], target_shape: list[int]) -> bool:
    """Return whether reshaping to ``target_shape`` is a pure squeeze/unsqueeze.

    Compares the two shapes with every size-1 dimension removed; the reshape
    is considered safe only if what's left is identical, in the same order.
    This rejects any reshape that would reorder, merge, or split a
    non-1-sized dimension -- ``torch.reshape`` itself would allow such a
    reshape as long as the total element count matches, even though the
    result could be silently wrong (e.g. transposed) rather than a shape-only
    correction.

    Each dimension is cast via ``int(d)`` (matching
    ``_build_fused_sample_input``'s own ``int(s)`` casts) rather than
    compared directly: under ``torch.jit.trace``, elements of
    ``tensor.shape`` are ``torch.Tensor`` instances, not plain ints, and
    comparing them with bare ``!=``/``==`` forces an implicit
    tensor-to-bool conversion that emits a ``TracerWarning``. The ``int(d)``
    cast still emits its own (expected, harmless) ``TracerWarning`` when
    ``source_shape`` comes from a traced tensor's actual ``.shape`` --
    unavoidable, since this check's entire purpose is validating the real
    runtime shape, not just the statically-known declared one. The decision
    this produces is safe to bake into the trace: a schema-declared
    feature's non-batch shape is invariant across calls by construction, so
    it can't legitimately differ between the traced example and any other
    real input for that same fused model.

    Args:
        source_shape: The value's actual shape (excluding batch).
        target_shape: The declared shape to reshape it to (excluding batch).

    Returns:
        ``True`` if the two shapes have the same sequence of non-1
        dimensions.
    """
    source_dims = [int(d) for d in source_shape if int(d) != 1]
    target_dims = [int(d) for d in target_shape if int(d) != 1]
    return source_dims == target_dims


class FusedModel(nn.Module):
    """Fuses a transform module and a predictor module with schema-driven merge.

    The native transform is always dict-in, dict-out: ``forward`` is called
    with a dict and returns a dict. The predictor's input is merged from the
    transform's output and passthrough values from the fused model's input;
    the predictor is then called with either a single dict
    (``predictor_takes_dict=True``) or positional tensor arguments in
    ``predictor_input_keys`` order (``predictor_takes_dict=False``).

    Attributes:
        transform_module: Native transform module.
        predictor_module: Predictor module.
        transform_input_keys: Input feature names fed to the transform.
        predictor_input_keys: Feature names the predictor expects, in order.
        predictor_takes_dict: Whether the predictor is called with a dict.
        predictor_input_shapes: Per-feature shape (excluding the batch
            dimension) the predictor's own schema declares, keyed by feature
            name. A feature present here has its value reshaped to
            ``[batch] + shape`` right before being handed to the predictor.
            A feature absent here (unknown/undeclared shape) is passed
            through unreshaped.
    """

    __constants__ = [  # noqa: RUF012
        "transform_input_keys",
        "predictor_input_keys",
        "predictor_takes_dict",
    ]

    def __init__(
        self,
        transform_module: nn.Module,
        predictor_module: nn.Module,
        transform_input_keys: list[str],
        predictor_input_keys: list[str],
        predictor_takes_dict: bool = False,
        predictor_input_shapes: dict[str, list[int]] | None = None,
    ) -> None:
        """Initialize the fused model.

        Args:
            transform_module: Native transform module. Always
                ``forward(inputs: dict) -> dict``.
            predictor_module: Predictor module. Its ``forward`` may accept a
                dict or multiple positional tensors.
            transform_input_keys: Input feature names fed to the transform
                (dict keys).
            predictor_input_keys: Feature names the predictor expects, in
                order.
            predictor_takes_dict: If ``True``, call the predictor with a
                dict; if ``False``, call it with positional tensors.
            predictor_input_shapes: Per-feature shape (excluding batch) from
                the predictor's own schema, keyed by feature name. Values are
                reshaped to this shape before being handed to the predictor,
                bridging any shape-convention mismatch between the transform
                that produced (or passed through) the value and the
                predictor's own declared expectation -- e.g.
                ``tabular_native_transform``'s scalar convention is shape
                ``[1]``, while ``tabular_trainer``'s ``ColumnConfig`` scalar
                convention is shape ``[]``. Omit an entry (or pass ``None``)
                to leave a feature unreshaped.
        """
        super().__init__()
        self.transform_module = transform_module
        self.predictor_module = predictor_module
        self.transform_input_keys = transform_input_keys
        self.predictor_input_keys = predictor_input_keys
        self.predictor_takes_dict = predictor_takes_dict
        self.predictor_input_shapes = predictor_input_shapes or {}

    def _reshape_for_predictor(self, key: str, value: torch.Tensor) -> torch.Tensor:
        """Reshape ``value`` to the predictor's own declared shape for ``key``.

        A feature absent from ``predictor_input_shapes`` (its schema item's
        ``shape`` was ``None``, i.e. genuinely unspecified -- see
        ``_schema_input_shapes``) is deliberately left unreshaped here,
        unlike ``_build_fused_sample_input``, which must still invent a
        concrete sample shape (defaulting to a single dimension of size 1)
        since it has to produce an actual zero-filled tensor for tracing.
        Reshaping a real value based on a guessed shape would risk silently
        corrupting data the schema never actually described; leaving it
        alone is the safer default when the target shape isn't known.

        Only a pure squeeze/unsqueeze of size-1 dimensions is performed (see
        ``_is_safe_reshape``) -- deliberately narrower than a bare
        ``value.reshape(...)``, which would happily "succeed" on any target
        shape with the same total element count, including one that
        silently transposes or scrambles genuinely multi-dimensional data
        rather than merely correcting a declared-shape convention mismatch.

        Args:
            key: Feature name.
            value: The value as produced by the transform or passed through
                from the fused model's input.

        Returns:
            ``value`` reshaped to ``[batch] + predictor_input_shapes[key]``
            when a shape is declared for ``key``; ``value`` unchanged
            otherwise.

        Raises:
            ValueError: If reshaping ``value`` to the declared shape would
                not be a pure squeeze/unsqueeze of size-1 dimensions --
                refusing rather than risking silently-wrong output.
            RuntimeError: If the reshape itself fails (e.g. a genuine
                element-count mismatch), re-raised with ``key`` and both
                shapes named so the failure doesn't require bisecting which
                of potentially several fused features it came from.
        """
        shape = self.predictor_input_shapes.get(key)
        if shape is None:
            return value
        source_shape = list(value.shape)
        target_shape = [value.shape[0], *shape]
        if not _is_safe_reshape(source_shape[1:], shape):
            raise ValueError(
                f"Refusing to reshape predictor input {key!r} from "
                f"{source_shape} to {target_shape}: this isn't a pure "
                "squeeze/unsqueeze of size-1 dimensions (non-1 dimensions "
                "differ), so reshaping could silently produce wrong "
                "results instead of just correcting a declared-shape "
                "convention mismatch."
            )
        try:
            return value.reshape(target_shape)
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to reshape predictor input {key!r} from "
                f"{source_shape} to {target_shape}: {e}"
            ) from e

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run the transform then the predictor, merging output with passthrough.

        Args:
            inputs: Named input tensors (e.g. batched features). Must contain
                all ``transform_input_keys`` and any ``predictor_input_keys``
                not produced by the transform.

        Returns:
            The predictor's output tensor.
        """
        # The native transform is always dict-in, dict-out.
        transform_in_dict = {k: inputs[k] for k in self.transform_input_keys}
        transformed_dict = self.transform_module(transform_in_dict)

        predictor_input_dict = {}
        for k in list(inputs.keys()):
            predictor_input_dict[k] = inputs[k]
        for k in list(transformed_dict.keys()):
            predictor_input_dict[k] = transformed_dict[k]

        if self.predictor_takes_dict:
            predictor_in_dict = {
                k: self._reshape_for_predictor(k, predictor_input_dict[k])
                for k in self.predictor_input_keys
            }
            return self.predictor_module(predictor_in_dict)
        predictor_parts = [
            self._reshape_for_predictor(k, predictor_input_dict[k])
            for k in self.predictor_input_keys
        ]
        return self.predictor_module(*predictor_parts)
