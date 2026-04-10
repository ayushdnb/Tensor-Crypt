import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
REQUIREMENTS = ROOT / "requirements.txt"
README = ROOT / "README.md"
PYGAME_CE_SPEC = "pygame-ce>=2.5.6,<2.6"


def _requirements_entries(path: Path) -> list[str]:
    entries = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            entries.append(line)
    return entries


def _pyproject_dependencies() -> list[str]:
    text = PYPROJECT.read_text(encoding="utf-8")
    match = re.search(r"dependencies\s*=\s*\[(?P<body>.*?)\]", text, re.DOTALL)
    assert match is not None, "pyproject.toml is missing a project.dependencies block"
    return re.findall(r'"([^"]+)"', match.group("body"))


def test_runtime_dependency_surfaces_require_pygame_ce():
    pyproject_dependencies = _pyproject_dependencies()
    requirements_dependencies = _requirements_entries(REQUIREMENTS)

    assert PYGAME_CE_SPEC in pyproject_dependencies
    assert PYGAME_CE_SPEC in requirements_dependencies
    assert "pygame" not in pyproject_dependencies
    assert "pygame" not in requirements_dependencies


def test_readme_install_instructions_explain_pygame_ce_namespace():
    readme = README.read_text(encoding="utf-8")

    assert "pygame-ce>=2.5.6,<2.6" in readme
    assert "`pygame-ce`" in readme
    assert "`pygame`" in readme
