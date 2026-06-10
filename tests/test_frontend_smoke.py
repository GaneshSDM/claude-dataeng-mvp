import py_compile
from pathlib import Path


def test_frontend_pages_compile():
    root = Path(__file__).resolve().parent.parent / "frontend"
    for f in list(root.glob("*.py")) + list((root / "pages").glob("*.py")):
        py_compile.compile(str(f), doraise=True)
