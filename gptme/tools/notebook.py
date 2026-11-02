"""
Gives the LLM agent the ability to read and edit cells in Jupyter notebooks (``.ipynb`` files).

A notebook is a JSON document, which is fragile to edit as text. This tool works on
cells instead: list them, replace one, insert one, delete one, or clear outputs.
Everything the tool does not touch (metadata, other cells, their outputs) is kept as is.
"""

import contextlib
import json
import os
import shutil
import tempfile
import uuid
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..message import Message
from ..util import path_with_tilde
from ..util.ask_execute import execute_with_confirmation
from .base import (
    Parameter,
    ToolSpec,
    ToolUse,
    file_newline,
    get_path,
)

OPERATIONS = ("list", "replace", "insert", "delete", "clear_outputs")
CELL_TYPES = ("code", "markdown", "raw")

# Limits for the `list` output, so a huge cell or output never floods the context.
MAX_LISTED_LINES_PER_CELL = 30
MAX_LISTED_CHARS_PER_CELL = 2000

instructions = """
Use this tool to read and edit Jupyter notebooks (`.ipynb`) cell by cell, instead of
editing the JSON with `patch` or `save`.

Cells are numbered from 0. Always `list` the notebook first, so you know the cell indices.
Indices shift after an `insert` or `delete`, so `list` again if you do several edits.

Operations:
- `list`: show every cell (index, type, source, and a summary of its outputs).
- `replace INDEX [TYPE]`: replace the source of a cell. Keeps the cell's id and metadata.
  Outputs of a code cell are cleared, since they would be stale. TYPE changes the cell type.
- `insert INDEX [TYPE]`: insert a new cell before INDEX (use the cell count to append).
  TYPE defaults to `code`.
- `delete INDEX`: delete a cell.
- `clear_outputs [INDEX]`: clear outputs and execution counts, of one cell or of all cells.

TYPE is one of `code`, `markdown` or `raw`. This tool edits the file only: it does not run cells.
""".strip()

instructions_format = {
    "markdown": """
The $PATH parameter MUST be on the same line as the code block start, not on the line after.

The first line of the block is the operation. Everything after the first line is the new cell source.

```notebook $PATH
OPERATION [INDEX] [TYPE]
$CELL_SOURCE
```
""".strip(),
    "tool": "Use the `operation`, `cell_index`, `cell_type` and `source` parameters.",
}


example_replace = "replace 1\ndf = load_data()\nprint(df.shape)\ndf.head()"
example_insert = "insert 2 markdown\n## Plot of sales over time"


def examples(tool_format):
    return f"""
> User: in `analysis.ipynb`, make the second cell also print the shape of the data
> Assistant: Let me look at the notebook first.
{ToolUse("notebook", ["analysis.ipynb"], "list").to_output(tool_format)}
> System: Notebook `analysis.ipynb`: 3 cells
[0] markdown
  # Sales analysis
[1] code exec=1 outputs=1 (stream)
  df = load_data()
  df.head()
[2] code
  df.plot()
> Assistant: Cell 1 loads the data. I'll update it.
{ToolUse("notebook", ["analysis.ipynb"], example_replace).to_output(tool_format)}
> System: Replaced cell 1 in `analysis.ipynb`

> User: add a markdown note before the plot
> Assistant:
{ToolUse("notebook", ["analysis.ipynb"], example_insert).to_output(tool_format)}
> System: Inserted markdown cell at index 2 in `analysis.ipynb`
""".strip()


# --- Parsing the request -------------------------------------------------------------


@dataclass(frozen=True)
class NotebookRequest:
    op: str
    index: int | None
    cell_type: str | None
    source: str


def parse_request(content: str) -> NotebookRequest:
    """Parse ``"OPERATION [INDEX] [TYPE]\\nSOURCE"`` into a request.

    Raises ValueError with a message the model can act on.
    """
    header, _, body = content.partition("\n")
    parts = header.split()
    if not parts:
        raise ValueError(
            "empty request: the first line must be an operation "
            f"({', '.join(OPERATIONS)})"
        )

    op = parts[0].lower()
    if op not in OPERATIONS:
        raise ValueError(
            f"unknown operation {parts[0]!r}, expected one of: {', '.join(OPERATIONS)}"
        )

    index: int | None = None
    cell_type: str | None = None
    for token in parts[1:]:
        if token.lower() in CELL_TYPES:
            if cell_type is not None:
                raise ValueError(f"cell type given twice: {header!r}")
            cell_type = token.lower()
            continue
        try:
            value = int(token)
        except ValueError:
            raise ValueError(
                f"unexpected argument {token!r}: expected a cell index (integer) "
                f"or a cell type ({', '.join(CELL_TYPES)})"
            ) from None
        if index is not None:
            raise ValueError(f"cell index given twice: {header!r}")
        index = value

    if op in ("replace", "insert", "delete") and index is None:
        raise ValueError(f"`{op}` needs a cell index, e.g. `{op} 0`")
    if op == "delete" and cell_type is not None:
        raise ValueError("`delete` does not take a cell type")

    # A single trailing newline is the end of the block, not part of the cell source.
    source = body.removesuffix("\n") if op in ("replace", "insert") else ""
    return NotebookRequest(op=op, index=index, cell_type=cell_type, source=source)


def content_from_kwargs(kwargs: dict[str, str]) -> str:
    """Build the markdown-style request from tool-format parameters."""
    header = " ".join(
        part
        for part in (
            kwargs.get("operation", ""),
            str(kwargs.get("cell_index", "")).strip(),
            kwargs.get("cell_type", ""),
        )
        if part
    )
    return f"{header}\n{kwargs.get('source', '')}"


# --- Reading and writing the notebook file -------------------------------------------


@dataclass(frozen=True)
class FileFormat:
    """How the file was formatted, so we can write it back the same way."""

    indent: int | str
    trailing_newline: bool


def _detect_indent(raw: str) -> int | str:
    lines = raw.split("\n")
    if len(lines) > 1:
        second = lines[1]
        spaces = len(second) - len(second.lstrip(" "))
        if spaces:
            return spaces
        if second.startswith("\t"):
            return "\t"
    return 1  # Jupyter's default


def resolve_notebook_path(path: Path) -> Path:
    """Resolve and validate the path of a notebook."""
    path_display = path
    path = path.expanduser().resolve()

    # Same rule as the patch tool: relative paths must stay inside the cwd.
    if not path_display.is_absolute():
        cwd = Path.cwd().resolve()
        try:
            path.relative_to(cwd)
        except ValueError as err:
            raise ValueError(
                f"Path traversal detected: {path_display} resolves to {path} "
                f"which is outside current directory {cwd}"
            ) from err

    if path.suffix != ".ipynb":
        raise ValueError(f"`{path.name}` is not a notebook (expected a .ipynb file)")
    if not path.exists():
        raise ValueError(f"No such file `{path}` (pwd: `{Path.cwd()}`)")
    if not path.is_file():
        raise ValueError(f"Cannot edit non-regular file `{path}`")
    return path


def load_notebook(path: Path) -> tuple[dict[str, Any], FileFormat]:
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise ValueError(f"`{path.name}` is not valid UTF-8") from None

    try:
        nb = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"`{path.name}` is not valid JSON: {e}") from None

    if not isinstance(nb, dict) or not isinstance(nb.get("cells"), list):
        raise ValueError(f"`{path.name}` is not a notebook: no `cells` list found")
    if nb.get("nbformat") != 4:
        raise ValueError(
            f"Unsupported notebook format (nbformat={nb.get('nbformat')!r}), "
            "only nbformat 4 is supported"
        )
    if not all(isinstance(c, dict) for c in nb["cells"]):
        raise ValueError(f"`{path.name}` has a malformed cell (expected an object)")

    return nb, FileFormat(
        indent=_detect_indent(raw),
        trailing_newline=raw.endswith("\n") or not raw,
    )


def save_notebook(path: Path, nb: dict[str, Any], fmt: FileFormat) -> None:
    """Write the notebook atomically: a failure never leaves a half-written file."""
    text = json.dumps(nb, indent=fmt.indent, ensure_ascii=False)
    if fmt.trailing_newline:
        text += "\n"
    # Raw newlines in the output are structural only: newlines in strings are escaped.
    newline = file_newline(path)
    if newline != "\n":
        text = text.replace("\n", newline)
    data = text.encode("utf-8")

    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        shutil.copymode(path, tmp_name)
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


# --- Cell helpers -----------------------------------------------------------------------


def split_source(source: str) -> list[str]:
    """Split source into the list-of-lines form Jupyter uses (lines keep their ``\\n``)."""
    if not source:
        return []
    lines = source.split("\n")
    result = [line + "\n" for line in lines[:-1]]
    if lines[-1]:
        result.append(lines[-1])
    return result


def source_text(cell: dict[str, Any]) -> str:
    """The cell source as one string (Jupyter allows a string or a list of lines)."""
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(str(line) for line in source)
    return str(source)


def _get_cell(nb: dict[str, Any], index: int | None) -> dict[str, Any]:
    cells = nb["cells"]
    if index is None or not 0 <= index < len(cells):
        if not cells:
            raise ValueError(
                f"cell index {index} out of range: the notebook has no cells"
            )
        raise ValueError(
            f"cell index {index} out of range: the notebook has {len(cells)} "
            f"cells (valid indices: 0..{len(cells) - 1})"
        )
    return cells[index]


def _make_cell_id(nb: dict[str, Any]) -> str:
    existing = {c.get("id") for c in nb["cells"]}
    while True:
        cell_id = uuid.uuid4().hex[:8]
        if cell_id not in existing:
            return cell_id


def _set_cell_type(cell: dict[str, Any], cell_type: str) -> None:
    cell["cell_type"] = cell_type
    if cell_type == "code":
        cell.setdefault("outputs", [])
        cell.setdefault("execution_count", None)
    else:
        # Only code cells may have outputs and an execution count.
        cell.pop("outputs", None)
        cell.pop("execution_count", None)


def _first_line(text: str, limit: int = 60) -> str:
    line = text.strip().split("\n", 1)[0]
    return line if len(line) <= limit else line[: limit - 1] + "…"


# --- Operations -----------------------------------------------------------------------------


def _op_replace(nb: dict[str, Any], req: NotebookRequest) -> str:
    cell = _get_cell(nb, req.index)
    old_type = cell.get("cell_type", "code")
    new_type = req.cell_type or old_type

    if new_type == old_type and req.source == source_text(cell):
        raise ValueError(
            f"replace did not change cell {req.index} "
            "(same source and type; use clear_outputs to only clear outputs)"
        )

    cell["source"] = split_source(req.source)
    _set_cell_type(cell, new_type)
    if new_type == "code":
        # The old outputs belong to the old source.
        cell["outputs"] = []
        cell["execution_count"] = None

    if new_type != old_type:
        return f"Replaced cell {req.index} (changed type {old_type} -> {new_type})"
    return f"Replaced cell {req.index}"


def _op_insert(nb: dict[str, Any], req: NotebookRequest) -> str:
    cells = nb["cells"]
    index = req.index
    # Unlike replace/delete, index == len(cells) is valid here: it appends.
    if index is None or not 0 <= index <= len(cells):
        raise ValueError(
            f"insert index {index} out of range: the notebook has {len(cells)} "
            f"cells (valid indices: 0..{len(cells)}, where {len(cells)} appends)"
        )

    cell_type = req.cell_type or "code"
    cell: dict[str, Any] = {"cell_type": cell_type, "metadata": {}}
    cell["source"] = split_source(req.source)
    _set_cell_type(cell, cell_type)
    # Notebooks from nbformat 4.5 on require a unique id on every cell.
    if (nb.get("nbformat_minor") or 0) >= 5:
        cell["id"] = _make_cell_id(nb)

    cells.insert(index, cell)
    return f"Inserted {cell_type} cell at index {index}"


def _op_delete(nb: dict[str, Any], req: NotebookRequest) -> str:
    _get_cell(nb, req.index)  # validates the index
    assert req.index is not None
    removed = nb["cells"].pop(req.index)
    preview = _first_line(source_text(removed))
    suffix = f" (`{preview}`)" if preview else " (empty)"
    return f"Deleted {removed.get('cell_type', 'code')} cell {req.index}{suffix}"


def _op_clear_outputs(nb: dict[str, Any], req: NotebookRequest) -> str:
    if req.index is not None:
        targets = [_get_cell(nb, req.index)]
    else:
        targets = nb["cells"]

    cleared = 0
    for cell in targets:
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs") or cell.get("execution_count") is not None:
            cell["outputs"] = []
            cell["execution_count"] = None
            cleared += 1

    if not cleared:
        raise ValueError("no outputs to clear")
    where = f"cell {req.index}" if req.index is not None else f"{cleared} cell(s)"
    return f"Cleared outputs of {where}"


_MUTATIONS = {
    "replace": _op_replace,
    "insert": _op_insert,
    "delete": _op_delete,
    "clear_outputs": _op_clear_outputs,
}


def _summarize_output(output: Any) -> str:
    """Name an output without printing it (outputs can be huge, e.g. base64 images)."""
    if not isinstance(output, dict):
        return "?"
    summary = str(output.get("output_type", "?"))
    data = output.get("data")
    if isinstance(data, dict) and data:
        summary += "[" + ",".join(sorted(data)) + "]"
    return summary


def format_listing(nb: dict[str, Any], path: Path) -> str:
    cells = nb["cells"]
    lines = [f"Notebook `{path_with_tilde(path)}`: {len(cells)} cells"]
    for i, cell in enumerate(cells):
        cell_type = cell.get("cell_type", "?")
        info = [f"[{i}] {cell_type}"]
        if cell.get("id"):
            info.append(f"id={cell['id']}")
        if cell_type == "code":
            if cell.get("execution_count") is not None:
                info.append(f"exec={cell['execution_count']}")
            outputs = cell.get("outputs") or []
            if outputs:
                kinds = ", ".join(_summarize_output(o) for o in outputs)
                info.append(f"outputs={len(outputs)} ({kinds})")
        lines.append(" ".join(info))

        text = source_text(cell)
        truncated = False
        if len(text) > MAX_LISTED_CHARS_PER_CELL:
            text, truncated = text[:MAX_LISTED_CHARS_PER_CELL], True
        src_lines = text.split("\n")
        if len(src_lines) > MAX_LISTED_LINES_PER_CELL:
            src_lines, truncated = src_lines[:MAX_LISTED_LINES_PER_CELL], True
        lines.extend(f"  {line}" for line in src_lines)
        if truncated:
            lines.append("  ... (cell truncated in this listing)")
    return "\n".join(lines)


# --- Tool entry points ---------------------------------------------------------------------


def execute_notebook_impl(
    content: str, path: Path | None
) -> Generator[Message, None, None]:
    """Actual implementation: load, apply one operation, save."""
    if path is None:
        raise ValueError("Path is required to edit a notebook")

    req = parse_request(content)
    resolved = resolve_notebook_path(path)
    nb, fmt = load_notebook(resolved)

    if req.op == "list":
        yield Message("system", format_listing(nb, resolved))
        return

    message = _MUTATIONS[req.op](nb, req)
    save_notebook(resolved, nb, fmt)
    yield Message("system", f"{message} in `{path_with_tilde(resolved)}`")


def preview_notebook(content: str, path: Path | None) -> str | None:
    """Validate the request before asking for confirmation, and show it as is."""
    parse_request(content)
    return content


def execute_notebook(
    code: str | None,
    args: list[str] | None,
    kwargs: dict[str, str] | None,
) -> Generator[Message, None, None]:
    """Run a notebook operation."""
    if code is None and kwargs is not None and "operation" in kwargs:
        code = content_from_kwargs(kwargs)

    if not code:
        yield Message(
            "system", "Error: No notebook operation provided by the assistant"
        )
        return

    # Listing changes nothing, so it does not need a confirmation.
    try:
        if parse_request(code).op == "list":
            yield from execute_notebook_impl(code, get_path(code, args, kwargs))
            return
    except ValueError as e:
        yield Message("system", f"Error during execution: {e}")
        return

    yield from execute_with_confirmation(
        code,
        args,
        kwargs,
        execute_fn=execute_notebook_impl,
        get_path_fn=get_path,
        preview_fn=preview_notebook,
        confirm_msg=f"Edit notebook {get_path(code, args, kwargs)}?",
        allow_edit=True,
    )


tool = ToolSpec(
    name="notebook",
    desc="Read and edit cells in Jupyter notebooks (.ipynb)",
    instructions=instructions,
    instructions_format=instructions_format,
    examples=examples,
    execute=execute_notebook,
    block_types=["notebook"],
    parameters=[
        Parameter(
            name="path",
            type="string",
            description="The path of the .ipynb notebook.",
            required=True,
        ),
        Parameter(
            name="operation",
            type="string",
            description="What to do with the notebook.",
            enum=list(OPERATIONS),
            required=True,
        ),
        Parameter(
            name="cell_index",
            type="integer",
            description=(
                "0-based cell index. Needed for replace, insert and delete; "
                "optional for clear_outputs."
            ),
        ),
        Parameter(
            name="cell_type",
            type="string",
            description="Cell type for replace or insert (default: code for insert).",
            enum=list(CELL_TYPES),
        ),
        Parameter(
            name="source",
            type="string",
            description="The new cell source, for replace and insert.",
        ),
    ],
    hints=frozenset({"file-ops", "destructive"}),
)
__doc__ = tool.get_doc(__doc__)
