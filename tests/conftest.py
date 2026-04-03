"""Shared pytest fixtures for neural-memory tests."""

import pytest
from pathlib import Path

from neural_memory.config import NeuralConfig, RedactionConfig
from neural_memory.models import IndexMode
from neural_memory.storage import Storage


SAMPLE_SOURCE = '''"""A sample module for testing."""

import os
from pathlib import Path


def greet(name: str) -> str:
    """Return a greeting string."""
    return f"Hello, {name}!"


def compute(x: int, y: int = 0) -> int:
    """Add two numbers together."""
    result = greet("world")
    return x + y


def _helper(val):
    """Private helper function."""
    if val > 0:
        return val
    return 0


class Animal:
    """An animal class."""

    def __init__(self, name: str):
        self.name = name

    def speak(self) -> str:
        """Make the animal speak."""
        return f"{self.name} says something"

    def _private_method(self):
        pass


class Dog(Animal):
    """A dog that inherits from Animal."""

    def speak(self) -> str:
        """Make the dog bark."""
        result = compute(1, 2)
        return f"{self.name} says woof"
'''


@pytest.fixture
def sample_source() -> str:
    return SAMPLE_SOURCE


@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    """Create a temporary project directory with sample Python files."""
    # Main module
    (tmp_path / "mymodule.py").write_text(SAMPLE_SOURCE, encoding="utf-8")

    # A second module for cross-file tests
    second = '''"""Second module."""

from mymodule import greet


def say_hello():
    """Call greet from mymodule."""
    return greet("test")
'''
    (tmp_path / "second.py").write_text(second, encoding="utf-8")

    return tmp_path


@pytest.fixture
def neural_config(tmp_project: Path) -> NeuralConfig:
    """Return a NeuralConfig pointing at tmp_project, with AST_ONLY mode."""
    return NeuralConfig(
        project_root=str(tmp_project),
        index_mode=IndexMode.AST_ONLY,
        include_patterns=["**/*.py"],
        exclude_patterns=["**/__pycache__/**"],
    )


@pytest.fixture
def storage(tmp_project: Path) -> Storage:
    """Open and yield a Storage instance, closing it after the test."""
    s = Storage(str(tmp_project))
    s.open()
    yield s
    s.close()
