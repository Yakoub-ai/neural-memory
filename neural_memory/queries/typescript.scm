; ── TypeScript tree-sitter queries for neural memory ─────────────────────────
; Extends JavaScript with TS-specific constructs

; ── Function declarations ──────────────────────────────────────────────────────
(function_declaration
  name: (identifier) @function.name
  parameters: (formal_parameters) @function.params
  return_type: (type_annotation)? @function.return_type
  body: (statement_block) @function.body
) @function.def

(export_statement
  declaration: (function_declaration
    name: (identifier) @function.name
    parameters: (formal_parameters) @function.params
    return_type: (type_annotation)? @function.return_type
  ) @function.def
)

; ── Arrow functions ────────────────────────────────────────────────────────────
(variable_declarator
  name: (identifier) @function.name
  value: (arrow_function
    parameters: (formal_parameters) @function.params
    return_type: (type_annotation)? @function.return_type
  )
) @function.arrow

; ── Method definitions ─────────────────────────────────────────────────────────
(method_definition
  name: (property_identifier) @method.name
  parameters: (formal_parameters) @method.params
  return_type: (type_annotation)? @method.return_type
  body: (statement_block) @method.body
) @method.def

; ── Classes ────────────────────────────────────────────────────────────────────
(class_declaration
  name: (type_identifier) @class.name
  (class_heritage
    (implements_clause
      (type_identifier) @class.implements
    )
  )?
  (class_heritage
    (extends_clause
      (identifier) @class.extends
    )
  )?
  body: (class_body) @class.body
) @class.def

; ── Interfaces ────────────────────────────────────────────────────────────────
(interface_declaration
  name: (type_identifier) @interface.name
  (extends_type_clause
    (type_identifier) @interface.extends
  )?
  body: (interface_body) @interface.body
) @interface.def

(export_statement
  declaration: (interface_declaration
    name: (type_identifier) @interface.name
  ) @interface.def
)

; ── Type aliases ──────────────────────────────────────────────────────────────
(type_alias_declaration
  name: (type_identifier) @type_alias.name
  value: (_) @type_alias.value
) @type_alias.def

(export_statement
  declaration: (type_alias_declaration
    name: (type_identifier) @type_alias.name
  ) @type_alias.def
)

; ── Enums ─────────────────────────────────────────────────────────────────────
(enum_declaration
  name: (identifier) @enum.name
  body: (enum_body) @enum.body
) @enum.def

(export_statement
  declaration: (enum_declaration
    name: (identifier) @enum.name
  ) @enum.def
)

; ── Imports ────────────────────────────────────────────────────────────────────
(import_statement
  source: (string) @import.source
) @import.stmt

(named_imports
  (import_specifier
    name: (identifier) @import.name
  )
) @import.named

; ── Calls ──────────────────────────────────────────────────────────────────────
(call_expression
  function: (identifier) @call.name
) @call.expr

(call_expression
  function: (member_expression
    property: (property_identifier) @call.name
  )
) @call.member_expr

; ── Decorators ────────────────────────────────────────────────────────────────
(decorator
  (identifier) @decorator.name
) @decorator.simple

(decorator
  (call_expression
    function: (identifier) @decorator.name
  )
) @decorator.call
