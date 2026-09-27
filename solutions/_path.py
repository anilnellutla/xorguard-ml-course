"""Make the lessons/ directory importable from any solution file."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "lessons"))
