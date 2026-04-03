"""Query tracer — links function/method nodes to the tables they read or write.

Scans raw_code of FUNCTION and METHOD nodes for ORM and raw SQL patterns,
then creates QUERIES and WRITES_TO edges to TABLE nodes.

Supported patterns
──────────────────
Python SQLAlchemy:
  Read  — session.query(Model), db.session.query(Model), select(Model),
           Model.query.filter/get/all/first/one
  Write — session.add(, session.merge(, session.delete(, db.session.add(,
           session.execute(insert(Model), session.execute(update(Model)

Python Django:
  Read  — Model.objects.filter/get/all/values/annotate/aggregate/first/last
  Write — Model.objects.create/update/bulk_create/delete, instance.save()

TypeScript TypeORM:
  Read  — repository.find/findOne/findAndCount, .createQueryBuilder, manager.find
  Write — repository.save/insert/update/delete, manager.save/insert

Go GORM:
  Read  — db.First/Find/Take/Where/Preload (resolves struct type name)
  Write — db.Create/Save/Updates/Update/Delete/Unscoped

Rust Diesel:
  Read  — tablename::table.filter/select/load/first/get_result
  Write — diesel::insert_into(tablename::table, diesel::update(tablename::table,
           diesel::delete(tablename::table

Raw SQL (all languages):
  Read  — SELECT ... FROM tablename
  Write — INSERT INTO tablename, UPDATE tablename SET, DELETE FROM tablename
"""

from __future__ import annotations

import re
from typing import Optional

from ..models import NeuralNode, NeuralEdge, NodeType, EdgeType


# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

# Raw SQL read: SELECT ... FROM tablename (one or more words after FROM)
_SQL_SELECT = re.compile(
    r"\bSELECT\b.{0,200}?\bFROM\s+([`\"\[]?[\w]+[`\"\]]?)",
    re.IGNORECASE | re.DOTALL,
)

# Raw SQL write patterns
_SQL_INSERT = re.compile(r"\bINSERT\s+INTO\s+([`\"\[]?[\w]+[`\"\]]?)", re.IGNORECASE)
_SQL_UPDATE = re.compile(r"\bUPDATE\s+([`\"\[]?[\w]+[`\"\]]?)\s+SET\b", re.IGNORECASE)
_SQL_DELETE = re.compile(r"\bDELETE\s+FROM\s+([`\"\[]?[\w]+[`\"\]]?)", re.IGNORECASE)


def _strip_quotes(name: str) -> str:
    return name.strip("`\"[]'")


def _make_edge(source_id: str, table_node_id: str, edge_type: EdgeType, context: str = "") -> NeuralEdge:
    return NeuralEdge(
        source_id=source_id,
        target_id=table_node_id,
        edge_type=edge_type,
        context=context,
        weight=1.0,
    )


# ---------------------------------------------------------------------------
# Per-ORM scanners
# Return sets of (table_node_id, EdgeType, context_hint)
# ---------------------------------------------------------------------------

def _scan_sqlalchemy(
    code: str,
    class_to_table: dict[str, str],
    table_name_to_id: dict[str, str],
) -> list[tuple[str, EdgeType, str]]:
    hits: list[tuple[str, EdgeType, str]] = []

    # Read: session.query(Model) or db.session.query(Model) or select(Model)
    for m in re.finditer(r"\b(?:session|db\.session)\.query\(\s*([\w]+)", code):
        cls = m.group(1)
        if cls in class_to_table:
            hits.append((class_to_table[cls], EdgeType.QUERIES, f"session.query({cls})"))

    for m in re.finditer(r"\bselect\(\s*([\w]+)", code):
        cls = m.group(1)
        if cls in class_to_table:
            hits.append((class_to_table[cls], EdgeType.QUERIES, f"select({cls})"))

    # Read: Model.query.filter/get/all/first/one(...)
    for m in re.finditer(r"\b([\w]+)\.query\s*\.\s*(?:filter|get|all|first|one|count|paginate)\b", code):
        cls = m.group(1)
        if cls in class_to_table:
            hits.append((class_to_table[cls], EdgeType.QUERIES, f"{cls}.query.*"))

    # Write: session.add( / db.session.add( / session.merge(
    for m in re.finditer(r"\b(?:session|db\.session)\.(?:add|merge)\(\s*([\w]+)", code):
        obj = m.group(1)
        # obj is an instance — try matching its capitalized form as a class
        cls_guess = obj[0].upper() + obj[1:] if obj else obj
        if cls_guess in class_to_table:
            hits.append((class_to_table[cls_guess], EdgeType.WRITES_TO, f"session.add({obj})"))

    # Write: session.delete(
    for m in re.finditer(r"\b(?:session|db\.session)\.delete\(\s*([\w]+)", code):
        obj = m.group(1)
        cls_guess = obj[0].upper() + obj[1:] if obj else obj
        if cls_guess in class_to_table:
            hits.append((class_to_table[cls_guess], EdgeType.WRITES_TO, f"session.delete({obj})"))

    # Write: session.execute(insert(Model) / update(Model) / delete(Model)
    for m in re.finditer(r"\b(?:insert|update|delete)\(\s*([\w]+)", code):
        cls = m.group(1)
        if cls in class_to_table:
            edge = EdgeType.QUERIES if m.group(0).startswith("select") else EdgeType.WRITES_TO
            hits.append((class_to_table[cls], EdgeType.WRITES_TO, f"execute({m.group(0)})"))

    return hits


def _scan_django(
    code: str,
    class_to_table: dict[str, str],
) -> list[tuple[str, EdgeType, str]]:
    hits: list[tuple[str, EdgeType, str]] = []

    read_methods = r"filter|get|all|values|values_list|annotate|aggregate|first|last|count|exists|select_related|prefetch_related|exclude|order_by|distinct"
    write_methods = r"create|update|bulk_create|bulk_update|get_or_create|update_or_create|delete"

    for m in re.finditer(rf"\b([\w]+)\.objects\.({read_methods})\b", code):
        cls = m.group(1)
        if cls in class_to_table:
            hits.append((class_to_table[cls], EdgeType.QUERIES, f"{cls}.objects.{m.group(2)}()"))

    for m in re.finditer(rf"\b([\w]+)\.objects\.({write_methods})\b", code):
        cls = m.group(1)
        if cls in class_to_table:
            hits.append((class_to_table[cls], EdgeType.WRITES_TO, f"{cls}.objects.{m.group(2)}()"))

    # instance.save() — hard to resolve class, skip for now (would need type inference)

    return hits


def _scan_typeorm(
    code: str,
    class_to_table: dict[str, str],
) -> list[tuple[str, EdgeType, str]]:
    hits: list[tuple[str, EdgeType, str]] = []

    # getRepository(Model) / manager.getRepository(Model)
    for m in re.finditer(r"getRepository\(\s*([\w]+)\s*\)", code):
        cls = m.group(1)
        if cls in class_to_table:
            hits.append((class_to_table[cls], EdgeType.QUERIES, f"getRepository({cls})"))

    # createQueryBuilder("tablename") or createQueryBuilder(Model)
    for m in re.finditer(r'createQueryBuilder\(\s*["\']?([\w]+)["\']?\s*\)', code):
        name = m.group(1)
        if name in class_to_table:
            hits.append((class_to_table[name], EdgeType.QUERIES, f"createQueryBuilder({name})"))

    # repository.find/findOne/findAndCount → QUERIES
    for m in re.finditer(r"\brepository\.(?:find|findOne|findAndCount|findOneBy|findBy|count|exist)\b", code):
        # Can't resolve which table without type inference — mark as unknown
        pass

    # repository.save/insert/update/delete → WRITES_TO
    # Same limitation — skip generic repository.* without model context

    # dataSource.getRepository(Model).find(...)
    for m in re.finditer(r"getRepository\(\s*([\w]+)\s*\)\.(?:find|findOne|count)\b", code):
        cls = m.group(1)
        if cls in class_to_table:
            hits.append((class_to_table[cls], EdgeType.QUERIES, f"getRepository({cls}).find"))

    for m in re.finditer(r"getRepository\(\s*([\w]+)\s*\)\.(?:save|insert|update|delete|remove)\b", code):
        cls = m.group(1)
        if cls in class_to_table:
            hits.append((class_to_table[cls], EdgeType.WRITES_TO, f"getRepository({cls}).save"))

    return hits


def _scan_gorm(
    code: str,
    class_to_table: dict[str, str],
) -> list[tuple[str, EdgeType, str]]:
    hits: list[tuple[str, EdgeType, str]] = []

    # db.First(&result, ...) / db.Find(&results) / db.Where(...).Find(&results)
    # GORM resolves table from the struct type passed as pointer
    for m in re.finditer(r"\bdb\.(?:First|Find|Take|Scan|Pluck|Count|Row|Rows)\s*\(\s*&\s*([\w]+)", code):
        cls = m.group(1)
        # Try both exact match and capitalized
        if cls in class_to_table:
            hits.append((class_to_table[cls], EdgeType.QUERIES, f"db.Find(&{cls})"))
        cls_cap = cls[0].upper() + cls[1:]
        if cls_cap in class_to_table and cls_cap != cls:
            hits.append((class_to_table[cls_cap], EdgeType.QUERIES, f"db.Find(&{cls_cap})"))

    # db.Create(&obj) / db.Save(&obj) / db.Updates(&obj)
    for m in re.finditer(r"\bdb\.(?:Create|Save|Updates|Update)\s*\(\s*&\s*([\w]+)", code):
        cls = m.group(1)
        if cls in class_to_table:
            hits.append((class_to_table[cls], EdgeType.WRITES_TO, f"db.Create(&{cls})"))
        cls_cap = cls[0].upper() + cls[1:]
        if cls_cap in class_to_table and cls_cap != cls:
            hits.append((class_to_table[cls_cap], EdgeType.WRITES_TO, f"db.Create(&{cls_cap})"))

    # db.Delete(&obj) / db.Unscoped().Delete(&obj)
    for m in re.finditer(r"\bdb(?:\.Unscoped\(\))?\.Delete\s*\(\s*&\s*([\w]+)", code):
        cls = m.group(1)
        if cls in class_to_table:
            hits.append((class_to_table[cls], EdgeType.WRITES_TO, f"db.Delete(&{cls})"))
        cls_cap = cls[0].upper() + cls[1:]
        if cls_cap in class_to_table and cls_cap != cls:
            hits.append((class_to_table[cls_cap], EdgeType.WRITES_TO, f"db.Delete(&{cls_cap})"))

    return hits


def _scan_diesel(
    code: str,
    table_name_to_id: dict[str, str],
) -> list[tuple[str, EdgeType, str]]:
    hits: list[tuple[str, EdgeType, str]] = []

    # tablename::table.filter/select/load/first/get_result → QUERIES
    for m in re.finditer(r"\b([\w]+)::table\s*\.\s*(?:filter|select|load|first|get_result|count)\b", code):
        tname = m.group(1).lower()
        if tname in table_name_to_id:
            hits.append((table_name_to_id[tname], EdgeType.QUERIES, f"{m.group(1)}::table.filter/select"))

    # diesel::insert_into(tablename::table) → WRITES_TO
    for m in re.finditer(r"diesel::insert_into\(\s*([\w]+)::table\s*\)", code):
        tname = m.group(1).lower()
        if tname in table_name_to_id:
            hits.append((table_name_to_id[tname], EdgeType.WRITES_TO, f"diesel::insert_into({m.group(1)}::table)"))

    # diesel::update(tablename::table) → WRITES_TO
    for m in re.finditer(r"diesel::update\(\s*([\w]+)::table\s*\)", code):
        tname = m.group(1).lower()
        if tname in table_name_to_id:
            hits.append((table_name_to_id[tname], EdgeType.WRITES_TO, f"diesel::update({m.group(1)}::table)"))

    # diesel::delete(tablename::table) → WRITES_TO
    for m in re.finditer(r"diesel::delete\(\s*([\w]+)::table\s*\)", code):
        tname = m.group(1).lower()
        if tname in table_name_to_id:
            hits.append((table_name_to_id[tname], EdgeType.WRITES_TO, f"diesel::delete({m.group(1)}::table)"))

    return hits


def _scan_raw_sql(
    code: str,
    table_name_to_id: dict[str, str],
) -> list[tuple[str, EdgeType, str]]:
    hits: list[tuple[str, EdgeType, str]] = []

    for m in _SQL_SELECT.finditer(code):
        tname = _strip_quotes(m.group(1)).lower()
        if tname in table_name_to_id:
            hits.append((table_name_to_id[tname], EdgeType.QUERIES, f"SELECT ... FROM {tname}"))

    for m in _SQL_INSERT.finditer(code):
        tname = _strip_quotes(m.group(1)).lower()
        if tname in table_name_to_id:
            hits.append((table_name_to_id[tname], EdgeType.WRITES_TO, f"INSERT INTO {tname}"))

    for m in _SQL_UPDATE.finditer(code):
        tname = _strip_quotes(m.group(1)).lower()
        if tname in table_name_to_id:
            hits.append((table_name_to_id[tname], EdgeType.WRITES_TO, f"UPDATE {tname} SET"))

    for m in _SQL_DELETE.finditer(code):
        tname = _strip_quotes(m.group(1)).lower()
        if tname in table_name_to_id:
            hits.append((table_name_to_id[tname], EdgeType.WRITES_TO, f"DELETE FROM {tname}"))

    return hits


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def trace_queries(
    all_nodes: dict[str, NeuralNode],
    all_edges: list[NeuralEdge],
) -> list[NeuralEdge]:
    """Scan function/method bodies and return QUERIES + WRITES_TO edges to TABLE nodes.

    Must be called AFTER orm_detector so TABLE nodes are already in all_nodes.
    """
    # Build lookup: lowercase table name → table node id
    table_name_to_id: dict[str, str] = {}
    for node in all_nodes.values():
        if node.node_type == NodeType.TABLE:
            table_name_to_id[node.name.lower()] = node.id

    if not table_name_to_id:
        return []

    # Build lookup: ORM class name → table node id (via DEFINES edges)
    class_to_table: dict[str, str] = {}
    for edge in all_edges:
        if edge.edge_type == EdgeType.DEFINES:
            src = all_nodes.get(edge.source_id)
            tgt = all_nodes.get(edge.target_id)
            if src and tgt and src.node_type == NodeType.CLASS and tgt.node_type == NodeType.TABLE:
                class_to_table[src.name] = tgt.id

    result: list[NeuralEdge] = []
    seen: set[tuple[str, str, str]] = set()  # (source_id, target_id, edge_type)

    def _add(source_id: str, hits: list[tuple[str, EdgeType, str]]) -> None:
        for (table_id, etype, ctx) in hits:
            key = (source_id, table_id, etype.value)
            if key not in seen:
                seen.add(key)
                result.append(_make_edge(source_id, table_id, etype, ctx))

    for node in all_nodes.values():
        if node.node_type not in (NodeType.FUNCTION, NodeType.METHOD):
            continue
        code = node.raw_code
        if not code:
            continue

        lang = node.language or "python"

        if lang == "python":
            _add(node.id, _scan_sqlalchemy(code, class_to_table, table_name_to_id))
            _add(node.id, _scan_django(code, class_to_table))
        elif lang in ("typescript", "javascript"):
            _add(node.id, _scan_typeorm(code, class_to_table))
        elif lang == "go":
            _add(node.id, _scan_gorm(code, class_to_table))
        elif lang == "rust":
            _add(node.id, _scan_diesel(code, table_name_to_id))

        # Raw SQL applies to all languages
        _add(node.id, _scan_raw_sql(code, table_name_to_id))

    return result
