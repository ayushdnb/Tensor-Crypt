from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUT_FILE = BASE_DIR / "codes" / "evolution.txt"

# Folders to ignore anywhere in the path.
IGNORE_DIRS = {
    "__pycache__",
    ".git",
    ".hg",
    ".svn",
    "venv",
    ".venv",
    "env",
    ".env",
    ".mypy_cache",
    ".pytest_cache",
    ".pytest_tmp",
    "build",
    "dist",
    "audit_tmp",
}


def should_ignore(path: Path) -> bool:
    return any(part in IGNORE_DIRS for part in path.parts)


def main() -> None:
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    py_files = sorted(
        (
            path
            for path in BASE_DIR.rglob("*.py")
            if path.is_file() and not should_ignore(path.relative_to(BASE_DIR))
        ),
        key=lambda path: str(path.relative_to(BASE_DIR)).lower(),
    )

    with OUT_FILE.open("w", encoding="utf-8", errors="replace") as out:
        for index, path in enumerate(py_files):
            try:
                code = path.read_text(encoding="utf-8", errors="replace")
            except Exception as exc:
                code = f"# [ERROR READING FILE: {exc}]\n"

            out.write(code)

            if index < len(py_files) - 1:
                if not code.endswith("\n"):
                    out.write("\n")
                out.write("\n")

    print(f"Done. Wrote raw code from {len(py_files)} .py files into:\n{OUT_FILE}")


if __name__ == "__main__":
    main()
