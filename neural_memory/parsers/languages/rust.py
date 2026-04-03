"""Rust language configuration for tree-sitter."""

from __future__ import annotations

from ...models import NodeType
from ..base import ContainerSpec, LanguageConfig


def get_rust_config() -> LanguageConfig:
    import tree_sitter_rust as tsrs
    return LanguageConfig(
        language_id="rust",
        extensions=[".rs"],
        language_fn=tsrs.language,
        definition_types={
            "function_item": NodeType.FUNCTION,
            "struct_item": NodeType.CLASS,
            "enum_item": NodeType.TYPE_DEF,
            "trait_item": NodeType.CLASS,
            "type_item": NodeType.TYPE_DEF,
            "static_item": NodeType.CONFIG,
            "const_item": NodeType.CONFIG,
            "macro_definition": NodeType.OTHER,
        },
        containers=[
            ContainerSpec(
                ts_type="impl_item",
                model_type=NodeType.CLASS,
                body_field="body",
                nested_type_map={"function_item": NodeType.METHOD},
                name_field="name",  # may be empty; falls back to "type" in _extract_container
            ),
        ],
        import_node_types=["use_declaration"],
    )
