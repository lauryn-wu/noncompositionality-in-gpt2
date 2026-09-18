"""Synchronize the selected notebook with the Python export's named sections."""

import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
STEM = "pie_usage_probe"


def sync_notebook(script_path, notebook_path):
    """Make one cleared code cell per section, retaining notebook metadata."""
    script = Path(script_path).read_text(encoding="utf-8")
    notebook_path = Path(notebook_path)
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    parts = re.split(r"^# %% (.+)\n", script, flags=re.MULTILINE)
    if len(parts) == 1:
        raise ValueError("No named sections found in the selected export")
    previous = {cell.get("id"): cell for cell in notebook["cells"]}
    cells = []
    for title, source in zip(parts[1::2], parts[2::2]):
        cell_id = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        cell = {
            "cell_type": "code",
            "execution_count": None,
            "id": cell_id,
            "metadata": previous.get(cell_id, {}).get("metadata", {}),
            "outputs": [],
            "source": (f"# {title}\n" + source.rstrip("\n")).splitlines(keepends=True),
        }
        cells.append(cell)
    notebook["cells"] = cells
    notebook_path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    return len(cells)


def main():
    count = sync_notebook(
        ROOT / "scripts" / f"{STEM}.py", ROOT / "notebooks" / f"{STEM}.ipynb"
    )
    print(f"Synchronized {count} notebook sections")


if __name__ == "__main__":
    main()
