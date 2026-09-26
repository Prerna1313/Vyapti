"""Keep the maintained guides' local links and Python imports in sync."""

import ast
import importlib
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
GUIDES = (
    "README.md",
    "USER_GUIDE.md",
    "API_REFERENCE.md",
    "DEVELOPMENT_GUIDE.md",
    "REPOSITORY_STATUS.md",
)


@pytest.mark.parametrize("guide", GUIDES)
def test_local_markdown_links_exist(guide):
    path = ROOT / guide
    text = path.read_text(encoding="utf-8")
    for target in re.findall(r"\]\(([^)]+)\)", text):
        destination = target.split("#", 1)[0]
        if not destination or "://" in destination or destination.startswith("mailto:"):
            continue
        assert (path.parent / destination).exists(), f"{guide}: missing {destination}"


@pytest.mark.parametrize("guide", GUIDES)
def test_python_example_imports_exist(guide):
    text = (ROOT / guide).read_text(encoding="utf-8")
    blocks = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
    for block in blocks:
        tree = ast.parse(block, filename=guide)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("vyapti_simulator"):
                module = importlib.import_module(node.module)
                for name in node.names:
                    assert hasattr(module, name.name), f"{guide}: {node.module}.{name.name} missing"
            elif isinstance(node, ast.Import):
                for name in node.names:
                    if name.name.startswith("vyapti_simulator"):
                        importlib.import_module(name.name)
