"""TypeScript and JavaScript language configurations for tree-sitter."""

from __future__ import annotations

from ...models import NodeType
from ..base import ContainerSpec, LanguageConfig


def _make_ts_config(
    language_id: str,
    extensions: list[str],
    language_fn,
) -> LanguageConfig:
    return LanguageConfig(
        language_id=language_id,
        extensions=extensions,
        language_fn=language_fn,
        definition_types={
            "function_declaration": NodeType.FUNCTION,
            "generator_function_declaration": NodeType.FUNCTION,
            "abstract_method_signature": NodeType.METHOD,
            "interface_declaration": NodeType.CLASS,
            "type_alias_declaration": NodeType.TYPE_DEF,
            "enum_declaration": NodeType.TYPE_DEF,
            "lexical_declaration": NodeType.OTHER,  # const x = () => ...
        },
        containers=[
            ContainerSpec(
                ts_type="class_declaration",
                model_type=NodeType.CLASS,
                body_field="body",
                nested_type_map={
                    "method_definition": NodeType.METHOD,
                    "public_field_definition": NodeType.OTHER,
                },
            ),
            ContainerSpec(
                ts_type="abstract_class_declaration",
                model_type=NodeType.CLASS,
                body_field="body",
                nested_type_map={
                    "method_definition": NodeType.METHOD,
                    "abstract_method_signature": NodeType.METHOD,
                },
            ),
        ],
        import_node_types=["import_statement"],
        wrapper_type="export_statement",
        wrapper_field="declaration",
    )


def get_typescript_config() -> LanguageConfig:
    import tree_sitter_typescript as tsts
    return _make_ts_config("typescript", [".ts", ".tsx"], tsts.language_typescript)


def get_tsx_config() -> LanguageConfig:
    import tree_sitter_typescript as tsts
    return _make_ts_config("typescript", [".tsx"], tsts.language_tsx)


def get_javascript_config() -> LanguageConfig:
    import tree_sitter_javascript as tsjs
    return LanguageConfig(
        language_id="javascript",
        extensions=[".js", ".jsx", ".mjs", ".cjs"],
        language_fn=tsjs.language,
        definition_types={
            "function_declaration": NodeType.FUNCTION,
            "generator_function_declaration": NodeType.FUNCTION,
        },
        containers=[
            ContainerSpec(
                ts_type="class_declaration",
                model_type=NodeType.CLASS,
                body_field="body",
                nested_type_map={"method_definition": NodeType.METHOD},
            ),
        ],
        import_node_types=["import_statement"],
        wrapper_type="export_statement",
        wrapper_field="declaration",
    )
