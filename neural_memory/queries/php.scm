; ── PHP tree-sitter queries for neural memory ────────────────────────────────

; ── Functions ──────────────────────────────────────────────────────────────────
(function_definition
  name: (name) @function.name
  parameters: (formal_parameters) @function.params
  body: (compound_statement) @function.body
) @function.def

; ── Classes ────────────────────────────────────────────────────────────────────
(class_declaration
  name: (name) @class.name
  (base_clause
    (qualified_name) @class.extends
  )?
  (class_interface_clause
    (qualified_name) @class.implements
  )?
  body: (declaration_list) @class.body
) @class.def

; ── Interfaces ────────────────────────────────────────────────────────────────
(interface_declaration
  name: (name) @interface.name
  (base_clause
    (qualified_name) @interface.extends
  )?
  body: (declaration_list) @interface.body
) @interface.def

; ── Traits ────────────────────────────────────────────────────────────────────
(trait_declaration
  name: (name) @trait.name
  body: (declaration_list) @trait.body
) @trait.def

; ── Methods ────────────────────────────────────────────────────────────────────
(method_declaration
  name: (name) @method.name
  parameters: (formal_parameters) @method.params
  body: (compound_statement) @method.body
) @method.def

; ── Use (namespace imports) ───────────────────────────────────────────────────
(namespace_use_declaration
  (namespace_use_clause
    (qualified_name) @import.name
  )
) @import.stmt

(namespace_use_declaration
  (namespace_use_clause
    (qualified_name) @import.name
    alias: (name) @import.alias
  )
) @import.aliased_stmt

; ── Require/include ───────────────────────────────────────────────────────────
(include_expression
  (string) @import.path
) @import.include

(require_expression
  (string) @import.path
) @import.require

; ── Calls ──────────────────────────────────────────────────────────────────────
(function_call_expression
  function: (name) @call.name
) @call.expr

(member_call_expression
  name: (name) @call.name
) @call.member_expr

(scoped_call_expression
  name: (name) @call.name
) @call.scoped_expr

; ── Constants ─────────────────────────────────────────────────────────────────
(const_declaration
  (const_element
    (name) @constant.name
  )
) @constant.def
