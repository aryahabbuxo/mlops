import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pinned(lines: list[str]) -> set[str]:
    return {line.strip().lower() for line in lines if line.strip() and not line.startswith("#")}


def pyproject_pins() -> set[str]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    extras = [pin for group in project["optional-dependencies"].values() for pin in group]
    return pinned(project["dependencies"] + extras)


def test_every_dependency_is_pinned_exactly():
    for pin in pyproject_pins():
        assert "==" in pin and not any(op in pin for op in (">=", "<=", "~=", "!=")), pin


def test_requirements_txt_and_pyproject_list_the_same_pins():
    requirements = pinned((ROOT / "requirements.txt").read_text().splitlines())
    assert requirements == pyproject_pins()
