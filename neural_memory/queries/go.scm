; ── Go tree-sitter queries for neural memory ─────────────────────────────────

; ── Functions ──────────────────────────────────────────────────────────────────
(function_declaration
  name: (identifier) @function.name
  parameters: (parameter_list) @function.params
  result: (_)? @function.return_type
  body: (block) @function.body
) @function.def

; ── Methods (functions with receivers) ────────────────────────────────────────
(method_declaration
  receiver: (parameter_list
    (parameter_declaration
      type: (type_identifier) @method.receiver_type
    )
  )
  name: (field_identifier) @method.name
  parameters: (parameter_list) @method.params
  result: (_)? @method.return_type
  body: (block) @method.body
) @method.def

; ── Structs ────────────────────────────────────────────────────────────────────
(type_declaration
  (type_spec
    name: (type_identifier) @struct.name
    type: (struct_type
      (field_declaration_list) @struct.fields
    )
  ) @struct.def
)

; ── Interfaces ────────────────────────────────────────────────────────────────
(type_declaration
  (type_spec
    name: (type_identifier) @interface.name
    type: (interface_type) @interface.body
  ) @interface.def
)

; ── Type aliases ──────────────────────────────────────────────────────────────
(type_declaration
  (type_spec
    name: (type_identifier) @type_alias.name
    type: (type_identifier) @type_alias.value
  ) @type_alias.def
)

; ── Imports ────────────────────────────────────────────────────────────────────
(import_declaration
  (import_spec
    path: (interpreted_string_literal) @import.path
  )
) @import.stmt

(import_declaration
  (import_spec_list
    (import_spec
      path: (interpreted_string_literal) @import.path
    )
  )
) @import.group_stmt

; ── Calls ──────────────────────────────────────────────────────────────────────
(call_expression
  function: (identifier) @call.name
) @call.expr

(call_expression
  function: (selector_expression
    field: (field_identifier) @call.name
  )
) @call.selector_expr

; ── Constants ─────────────────────────────────────────────────────────────────
(const_declaration
  (const_spec
    name: (identifier) @constant.name
    value: (_)? @constant.value
  )
) @constant.def
