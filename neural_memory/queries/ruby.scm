; ── Ruby tree-sitter queries for neural memory ───────────────────────────────

; ── Method definitions ─────────────────────────────────────────────────────────
(method
  name: (identifier) @function.name
  parameters: (method_parameters)? @function.params
  body: (body_statement) @function.body
) @function.def

(singleton_method
  name: (identifier) @function.name
  parameters: (method_parameters)? @function.params
) @function.singleton_def

; ── Classes ────────────────────────────────────────────────────────────────────
(class
  name: (constant) @class.name
  superclass: (superclass
    (constant) @class.superclass
  )?
  body: (body_statement) @class.body
) @class.def

; ── Modules ────────────────────────────────────────────────────────────────────
(module
  name: (constant) @module.name
  body: (body_statement) @module.body
) @module.def

; ── Methods inside classes ────────────────────────────────────────────────────
(class
  name: (constant) @method.class_name
  body: (body_statement
    (method
      name: (identifier) @method.name
      parameters: (method_parameters)? @method.params
    ) @method.def
  )
)

; ── Requires (imports) ─────────────────────────────────────────────────────────
(call
  method: (identifier) @_req
  arguments: (argument_list (string) @import.path)
  (#match? @_req "^require(_relative)?$")
) @import.stmt

; ── Calls ──────────────────────────────────────────────────────────────────────
(call
  method: (identifier) @call.name
) @call.expr

(call
  receiver: (_)
  method: (identifier) @call.name
) @call.receiver_expr

; ── Constants (module-level) ──────────────────────────────────────────────────
(assignment
  left: (constant) @constant.name
  right: (_) @constant.value
) @constant.def
