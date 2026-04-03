"""Shared dataclasses for database schema awareness."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ColumnDef:
    """Describes a single column in a database table."""
    name: str
    col_type: str = ""
    is_primary: bool = False
    is_nullable: bool = True
    foreign_key: str = ""   # "other_table.col" or ""


@dataclass
class TableSchema:
    """Describes a single database table with its columns."""
    table_name: str
    columns: list[ColumnDef] = field(default_factory=list)
    source_node_id: str = ""   # ID of the CLASS node that defines this table
    language: str = ""
    file_path: str = ""
