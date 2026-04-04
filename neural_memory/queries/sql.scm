; ── SQL tree-sitter queries for neural memory ────────────────────────────────
; Note: uses tree-sitter-sql grammar which covers most SQL dialects

; ── Tables ────────────────────────────────────────────────────────────────────
(create_table_statement
  name: (object_reference
    name: (identifier) @table.name
  )
  columns: (column_definitions) @table.columns
) @table.def

(create_table_statement
  name: (object_reference
    schema: (identifier) @table.schema
    name: (identifier) @table.name
  )
) @table.qualified_def

; ── Views ─────────────────────────────────────────────────────────────────────
(create_view_statement
  name: (object_reference
    name: (identifier) @view.name
  )
  definition: (_) @view.query
) @view.def

; ── Functions ─────────────────────────────────────────────────────────────────
(create_function_statement
  name: (object_reference
    name: (identifier) @function.name
  )
) @function.def

; ── Stored procedures ─────────────────────────────────────────────────────────
(create_procedure_statement
  name: (object_reference
    name: (identifier) @procedure.name
  )
) @procedure.def

; ── Indexes ────────────────────────────────────────────────────────────────────
(create_index_statement
  name: (identifier) @index.name
  table: (object_reference
    name: (identifier) @index.table_name
  )
) @index.def

; ── Foreign key references (cross-table edges) ────────────────────────────────
(foreign_key_clause
  table: (object_reference
    name: (identifier) @fk.references_table
  )
) @fk.def

; ── Table references in queries (USES edges) ──────────────────────────────────
(from_clause
  (from_item
    (object_reference
      name: (identifier) @ref.table_name
    )
  )
) @ref.from

(join_clause
  table: (object_reference
    name: (identifier) @ref.table_name
  )
) @ref.join
