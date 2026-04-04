; ── Rust tree-sitter queries for neural memory ───────────────────────────────

; ── Functions ──────────────────────────────────────────────────────────────────
(function_item
  name: (identifier) @function.name
  parameters: (parameters) @function.params
  return_type: (type_identifier)? @function.return_type
  body: (block) @function.body
) @function.def

(function_item
  (visibility_modifier) @function.vis
  name: (identifier) @function.name
  parameters: (parameters) @function.params
  return_type: (_)? @function.return_type
) @function.pub_def

; ── Structs ────────────────────────────────────────────────────────────────────
(struct_item
  name: (type_identifier) @struct.name
  body: (_)? @struct.body
) @struct.def

(attribute_item) @struct.attr

; ── Enums ─────────────────────────────────────────────────────────────────────
(enum_item
  name: (type_identifier) @enum.name
  body: (enum_variant_list) @enum.body
) @enum.def

; ── Traits ────────────────────────────────────────────────────────────────────
(trait_item
  name: (type_identifier) @trait.name
  body: (declaration_list) @trait.body
) @trait.def

; ── Impl blocks (methods) ─────────────────────────────────────────────────────
(impl_item
  body: (declaration_list
    (function_item
      name: (identifier) @method.name
      parameters: (parameters) @method.params
      body: (block) @method.body
    ) @method.def
  )
)

; ── Type aliases ──────────────────────────────────────────────────────────────
(type_item
  name: (type_identifier) @type_alias.name
  type: (_) @type_alias.value
) @type_alias.def

; ── Use declarations (imports) ─────────────────────────────────────────────────
(use_declaration
  argument: (_) @import.path
) @import.stmt

; ── Module declarations ────────────────────────────────────────────────────────
(mod_item
  name: (identifier) @module.name
) @module.decl

; ── Calls ──────────────────────────────────────────────────────────────────────
(call_expression
  function: (identifier) @call.name
) @call.expr

(call_expression
  function: (field_expression
    field: (field_identifier) @call.name
  )
) @call.field_expr

(call_expression
  function: (scoped_identifier
    name: (identifier) @call.name
  )
) @call.scoped_expr

; ── Attributes (like decorators) ──────────────────────────────────────────────
(attribute_item
  (attribute
    (identifier) @attribute.name
  )
) @attribute.item

; ── Constants ─────────────────────────────────────────────────────────────────
(const_item
  name: (identifier) @constant.name
  type: (_) @constant.type
  value: (_) @constant.value
) @constant.def
