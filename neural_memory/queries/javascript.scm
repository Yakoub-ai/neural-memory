; ── JavaScript tree-sitter queries for neural memory ─────────────────────────

; ── Function declarations ──────────────────────────────────────────────────────
(function_declaration
  name: (identifier) @function.name
  parameters: (formal_parameters) @function.params
  body: (statement_block) @function.body
) @function.def

(export_statement
  declaration: (function_declaration
    name: (identifier) @function.name
    parameters: (formal_parameters) @function.params
  ) @function.def
)

; ── Arrow functions assigned to variables ─────────────────────────────────────
(variable_declarator
  name: (identifier) @function.name
  value: (arrow_function
    parameters: (formal_parameters) @function.params
    body: (_) @function.body
  )
) @function.arrow

(lexical_declaration
  (variable_declarator
    name: (identifier) @function.name
    value: (arrow_function
      parameters: (formal_parameters) @function.params
    )
  )
) @function.arrow_decl

; ── Method definitions ─────────────────────────────────────────────────────────
(method_definition
  name: (property_identifier) @method.name
  parameters: (formal_parameters) @method.params
  body: (statement_block) @method.body
) @method.def

; ── Classes ────────────────────────────────────────────────────────────────────
(class_declaration
  name: (identifier) @class.name
  (class_heritage
    (identifier) @class.extends
  )?
  body: (class_body) @class.body
) @class.def

(export_statement
  declaration: (class_declaration
    name: (identifier) @class.name
  ) @class.def
)

; ── Imports ────────────────────────────────────────────────────────────────────
(import_statement
  source: (string) @import.source
) @import.stmt

(import_clause
  (identifier) @import.default
) @import.default_clause

(named_imports
  (import_specifier
    name: (identifier) @import.name
  )
) @import.named

; ── Requires (CommonJS) ────────────────────────────────────────────────────────
(call_expression
  function: (identifier) @_require
  arguments: (arguments (string) @import.source)
  (#eq? @_require "require")
) @import.require

; ── Calls ──────────────────────────────────────────────────────────────────────
(call_expression
  function: (identifier) @call.name
) @call.expr

(call_expression
  function: (member_expression
    property: (property_identifier) @call.name
  )
) @call.member_expr

; ── Exports ────────────────────────────────────────────────────────────────────
(export_statement
  declaration: (variable_declaration
    (variable_declarator
      name: (identifier) @export.name
    )
  )
) @export.stmt
