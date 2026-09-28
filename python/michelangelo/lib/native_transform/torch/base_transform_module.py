"""The executable ``TorchTransformModule`` DAG runner.

Converts a fitted
:class:`~michelangelo.lib.native_transform.torch.transform_spec.TransformSpec`
into a single ``torch.nn.Module`` that runs its layers in topological order.
The module is TorchScript-exportable, so the exact same transform graph runs
at training time (batched, ahead of the model) and at serving time (embedded
in the model artifact). ``load_transform_module_from_spec_dict`` reverses the
serialization side of that same round trip, rebuilding a ``TransformSpec``
(and materializing its ``TorchTransformModule``) from a
``TransformSpec.to_dict()`` dict -- used as a fused model's Hydra
reconstruction factory (see
:mod:`michelangelo.lib.shared.utils.model_fuser._private.fuse`).
"""

from __future__ import annotations

from typing import Any

import torch

from michelangelo.lib.native_transform.torch.base_layers import TorchTransformBaseLayer
from michelangelo.lib.native_transform.torch.transform_spec import TransformSpec
from michelangelo.lib.native_transform.torch.utils import generate_layer_name

__all__ = [
    "TorchTransformModule",
    "get_transform_module",
    "load_transform_module_from_spec_dict",
]


class TorchTransformModule(torch.nn.Module):
    """Executes a DAG of transform layers, in topological order.

    Args:
        name: The module's name.
        input_cols: Column names the module expects as input.
        output_cols: Column names the module returns as output.
        layers: The transform layers to run, already in topological order.
            Every element must be a
            :class:`~michelangelo.lib.native_transform.torch.base_layers.TorchTransformBaseLayer`.

    Raises:
        AssertionError: If any element of ``layers`` is not a
            ``TorchTransformBaseLayer``.
    """

    def __init__(
        self,
        name: str,
        input_cols: list[str],
        output_cols: list[str],
        layers: torch.nn.ModuleList,
    ) -> None:
        """Initialize the TorchTransformModule.

        Args:
            name: The module's name.
            input_cols: Column names the module expects as input.
            output_cols: Column names the module returns as output.
            layers: The transform layers to run, already in topological
                order. Every element must be a ``TorchTransformBaseLayer``.

        Raises:
            AssertionError: If any element of ``layers`` is not a
                ``TorchTransformBaseLayer``.
        """
        super().__init__()
        assert all(isinstance(m, TorchTransformBaseLayer) for m in layers), (
            "All modules must be instances of TorchTransformBaseLayer"
        )
        self.name = name
        self.input_cols = input_cols
        self.output_cols = output_cols
        self.layers = layers

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run every layer in order, threading outputs into later layers' inputs.

        Args:
            inputs: Mapping from column name to tensor; must contain every
                column in ``self.input_cols``.

        Returns:
            A mapping from each column in ``self.output_cols`` to its
            computed tensor.

        Raises:
            ValueError: If ``inputs`` is missing a declared input column, or
                a layer's declared input column is not yet available (i.e.
                ``layers`` is not in a valid topological order).
        """
        # Node registry stores inputs and intermediate outputs from each layer.
        nodes: dict[str, torch.Tensor] = {}
        for name in self.input_cols:
            if name not in inputs:
                raise ValueError(f"Missing input name {name}. Inputs={inputs}")
            nodes[name] = inputs[name]

        for layer in self.layers:
            layer_inputs: dict[str, torch.Tensor] = {}
            for col in layer.input_cols:
                if col not in nodes:
                    raise ValueError(
                        f"Missing input name {col} for layer {layer.name}."
                    )
                layer_inputs[col] = nodes[col]
            layer_outputs = layer(layer_inputs)
            nodes.update(layer_outputs)

        results: dict[str, torch.Tensor] = {}
        for output_col in self.output_cols:
            results[output_col] = nodes[output_col]
        return results


def get_transform_module(
    transform_spec: TransformSpec,
    start_level: int,
    end_level: int | None = None,
    output_cols: set[str] | None = None,
) -> TorchTransformModule | None:
    """Materialize a level range of a ``TransformSpec`` into a ``TorchTransformModule``.

    Args:
        transform_spec: The fitted spec DAG to materialize.
        start_level: The first transform level to include (inclusive).
        end_level: The last transform level to include (inclusive). Defaults
            to the spec's maximum level; clamped to it if given a larger
            value.
        output_cols: The columns the module should return. Defaults to every
            output column produced across the included levels.

    Returns:
        The materialized module, or ``None`` if the level range contains no
        layers.
    """
    layers = []
    transform_input_cols: set[str] = set()
    transform_output_cols: set[str] = set()
    end_level = (
        transform_spec.get_max_transform_level()
        if end_level is None
        else min(end_level, transform_spec.get_max_transform_level())
    )
    for level in range(start_level, end_level + 1):
        layers.extend(transform_spec.to_transform_layers(level))
        transform_input_cols.update(transform_spec.get_transform_input_cols(level))
        transform_output_cols.update(transform_spec.get_transform_output_cols(level))
    if len(layers) == 0:
        return None
    input_cols = transform_input_cols - transform_output_cols
    return TorchTransformModule(
        name=generate_layer_name(TorchTransformModule.__name__.lower()),
        input_cols=sorted(input_cols),
        output_cols=sorted(transform_output_cols)
        if output_cols is None
        else sorted(output_cols),
        layers=torch.nn.ModuleList(layers),
    )


def load_transform_module_from_spec_dict(
    spec_dict: dict[str, Any],
    start_level: int = 0,
    end_level: int | None = None,
) -> TorchTransformModule:
    """Reconstruct a ``TorchTransformModule`` from a ``TransformSpec.to_dict()`` dict.

    Used as a Hydra ``_target_`` factory (see
    :mod:`michelangelo.lib.shared.utils.model_fuser._private.fuse`) to
    materialize a fitted native-transform module from its serialized spec at
    model-load time. ``TorchTransformModule`` cannot be built directly from
    ``to_dict()``'s output -- that dict has spec-DAG shape, not
    ``TorchTransformModule.__init__``'s ``name``/``input_cols``/
    ``output_cols``/``layers`` constructor shape.

    Args:
        spec_dict: A dict produced by ``TransformSpec.to_dict()``.
        start_level: The first transform level to include (inclusive).
        end_level: The last transform level to include (inclusive), or
            ``None`` for the spec's maximum level.

    Returns:
        The materialized ``TorchTransformModule``.

    Raises:
        ValueError: If ``[start_level, end_level]`` contains no layers --
            a factory used as a Hydra ``_target_`` must return a module, not
            ``None``.
    """
    transform_spec = TransformSpec(raw_transform_specs={"transform_specs": []})
    transform_spec.load_from_dict(spec_dict)
    module = get_transform_module(transform_spec, start_level, end_level)
    if module is None:
        resolved_end_level = (
            transform_spec.get_max_transform_level() if end_level is None else end_level
        )
        raise ValueError(
            f"No transform layers found in levels [{start_level}, "
            f"{resolved_end_level}]; load_transform_module_from_spec_dict "
            "cannot materialize an empty TorchTransformModule."
        )
    return module
