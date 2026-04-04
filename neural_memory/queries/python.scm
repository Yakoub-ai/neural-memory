; ── Python tree-sitter queries for neural memory ─────────────────────────────
; Named captures drive node/edge extraction in ts_parser.py

; ── Functions ──────────────────────────────────────────────────────────────────
(function_definition
  name: (identifier) @function.name
  parameters: (parameters) @function.params
  return_type: (type)? @function.return_type
  body: (block
    (expression_statement (string) @function.docstring)?
  )
) @function.def

(decorated_definition
  (decorator) @function.decorator
  definition: (function_definition
    name: (identifier) @function.name
  )
) @function.decorated

; ── Async functions ────────────────────────────────────────────────────────────
(function_definition
  "async" @function.async
  name: (identifier) @function.name
  parameters: (parameters) @function.params
  body: (block
    (expression_statement (string) @function.docstring)?
  )
) @function.def

; ── Classes ────────────────────────────────────────────────────────────────────
(class_definition
  name: (identifier) @class.name
  superclasses: (argument_list)? @class.bases
  body: (block
    (expression_statement (string) @class.docstring)?
  )
) @class.def

(decorated_definition
  (decorator) @class.decorator
  definition: (class_definition
    name: (identifier) @class.name
  )
) @class.decorated

; ── Methods (functions nested inside a class body) ────────────────────────────
(class_definition
  name: (identifier) @method.class_name
  body: (block
    (function_definition
      name: (identifier) @method.name
      parameters: (parameters) @method.params
      body: (block
        (expression_statement (string) @method.docstring)?
      )
    ) @method.def
  )
)

; ── Imports ────────────────────────────────────────────────────────────────────
(import_statement
  name: (dotted_name) @import.module
) @import.stmt

(import_from_statement
  module_name: (dotted_name) @import.from_module
  name: (dotted_name) @import.name
) @import.from_stmt

(import_from_statement
  (wildcard_import) @import.wildcard
) @import.from_stmt

; ── Calls ──────────────────────────────────────────────────────────────────────
(call
  function: (identifier) @call.name
) @call.expr

(call
  function: (attribute
    attribute: (identifier) @call.name
  )
) @call.expr

; ── Module-level constants ─────────────────────────────────────────────────────
(expression_statement
  (assignment
    left: (identifier) @constant.name
    right: (_) @constant.value
  )
) @constant.stmt
