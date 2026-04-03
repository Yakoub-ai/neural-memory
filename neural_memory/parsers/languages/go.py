"""Go language configuration for tree-sitter."""

from __future__ import annotations

from ...models import NodeType
from ..base import ContainerSpec, LanguageConfig, TypeSpecConfig


def get_go_config() -> LanguageConfig:
    import tree_sitter_go as tsgo
    return LanguageConfig(
        language_id="go",
        extensions=[".go"],
        language_fn=tsgo.language,
        definition_types={
            # Top-level functions (no receiver)
            "function_declaration": NodeType.FUNCTION,
            # Methods with a receiver — treated as METHOD at module level;
            # they'll be grouped under their receiver type in the graph via naming
            "method_declaration": NodeType.METHOD,
        },
        definition_name_fields={
            "method_declaration": "name",
        },
        containers=[],  # Go has no class bodies — methods are top-level
        import_node_types=["import_declaration"],
        # Go-style type declarations (struct, interface, type alias)
        type_spec=TypeSpecConfig(
            outer_type="type_declaration",
            spec_type="type_spec",
            name_field="name",
            value_field="type",
            value_type_map={
                "struct_type": NodeType.CLASS,
                "interface_type": NodeType.CLASS,
                "pointer_type": NodeType.TYPE_DEF,
                "slice_type": NodeType.TYPE_DEF,
                "map_type": NodeType.TYPE_DEF,
                "channel_type": NodeType.TYPE_DEF,
                "function_type": NodeType.TYPE_DEF,
                "qualified_type": NodeType.TYPE_DEF,
                "type_identifier": NodeType.TYPE_DEF,
            },
        ),
    )
