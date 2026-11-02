import copy
import json
from pathlib import Path

import pytest

from gptme.tools.notebook import (
    execute_notebook,
    execute_notebook_impl,
    parse_request,
    source_text,
    split_source,
)


def make_notebook(minor: int = 5) -> dict:
    """A small but realistic nbformat 4 notebook."""
    cells: list[dict] = [
        {
            "cell_type": "markdown",
            "id": "md000001",
            "metadata": {"tags": ["intro"]},
            "source": ["# Title\n", "Some text"],
        },
        {
            "cell_type": "code",
            "id": "code0001",
            "metadata": {},
            "execution_count": 1,
            "outputs": [
                {"output_type": "stream", "name": "stdout", "text": ["hello\n"]},
                {
                    "output_type": "display_data",
                    "data": {"image/png": "iVBORw0KGgo=", "text/plain": ["<Figure>"]},
                    "metadata": {},
                },
            ],
            "source": ["print('hello')\n", "x = 1"],
        },
        {
            "cell_type": "code",
            "id": "code0002",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": ["y = 2"],
        },
    ]
    if minor < 5:
        for cell in cells:
            del cell["id"]
    return {
        "cells": cells,
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}},
        "nbformat": 4,
        "nbformat_minor": minor,
    }


def write_nb(path: Path, nb: dict, indent: int = 1) -> Path:
    path.write_text(json.dumps(nb, indent=indent, ensure_ascii=False) + "\n")
    return path


def read_nb(path: Path) -> dict:
    return json.loads(path.read_text())


def run(path, code: str) -> str:
    """Run a notebook operation, return the text of all messages."""
    return "\n".join(m.content for m in execute_notebook(code, [str(path)], None))


@pytest.fixture
def nb_path(tmp_path) -> Path:
    return write_nb(tmp_path / "test.ipynb", make_notebook())


# --- parsing ---------------------------------------------------------------------------


def test_parse_replace_with_type_and_source():
    req = parse_request("replace 2 markdown\n# Hi\nthere\n")
    assert (req.op, req.index, req.cell_type) == ("replace", 2, "markdown")
    assert req.source == "# Hi\nthere"


def test_parse_list_and_clear_outputs():
    assert parse_request("list").op == "list"
    assert parse_request("clear_outputs").index is None
    assert parse_request("clear_outputs 1").index == 1


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "frobnicate 1",
        "replace",
        "delete",
        "replace one",
        "replace 1 2",
        "delete 1 code",
    ],
)
def test_parse_errors(bad):
    with pytest.raises(ValueError, match=r"\S"):
        parse_request(bad)


def test_split_source_roundtrip():
    for text in ["", "a", "a\n", "a\nb", "a\n\nb\n", "\n", "x = 'é ✓'\n"]:
        assert "".join(split_source(text)) == text
    assert split_source("a\nb") == ["a\n", "b"]


# --- list -------------------------------------------------------------------------------


def test_list_shows_cells_and_summarizes_outputs(nb_path):
    out = run(nb_path, "list")
    assert "3 cells" in out
    assert "[0] markdown id=md000001" in out
    assert "[1] code id=code0001 exec=1 outputs=2" in out
    assert "display_data[image/png,text/plain]" in out
    assert "  print('hello')" in out
    # the image data itself must never be printed
    assert "iVBORw0KGgo" not in out


def test_list_truncates_huge_cells(tmp_path):
    nb = make_notebook()
    nb["cells"][2]["source"] = [f"line {i}\n" for i in range(500)]
    path = write_nb(tmp_path / "big.ipynb", nb)
    out = run(path, "list")
    assert "line 29" in out
    assert "line 30" not in out
    assert "truncated" in out


def test_list_does_not_modify_file(nb_path):
    before = nb_path.read_bytes()
    run(nb_path, "list")
    assert nb_path.read_bytes() == before


# --- replace ----------------------------------------------------------------------------


def test_replace_source_clears_stale_outputs(nb_path):
    out = run(nb_path, "replace 1\nprint('new')\nz = 3")
    assert "Replaced cell 1" in out
    cell = read_nb(nb_path)["cells"][1]
    assert source_text(cell) == "print('new')\nz = 3"
    assert cell["outputs"] == []
    assert cell["execution_count"] is None
    # id and metadata of the cell are kept
    assert cell["id"] == "code0001"


def test_replace_leaves_everything_else_untouched(nb_path):
    before = read_nb(nb_path)
    run(nb_path, "replace 1\nprint('new')")
    after = read_nb(nb_path)

    expected = copy.deepcopy(before)
    expected["cells"][1].update(
        {"source": ["print('new')"], "outputs": [], "execution_count": None}
    )
    assert after == expected


def test_replace_changes_cell_type_markdown_to_code(nb_path):
    out = run(nb_path, "replace 0 code\nprint(1)")
    assert "markdown -> code" in out
    cell = read_nb(nb_path)["cells"][0]
    assert cell["cell_type"] == "code"
    assert cell["outputs"] == [] and cell["execution_count"] is None


def test_replace_changes_cell_type_code_to_markdown(nb_path):
    run(nb_path, "replace 1 markdown\n# now text")
    cell = read_nb(nb_path)["cells"][1]
    assert cell["cell_type"] == "markdown"
    assert "outputs" not in cell and "execution_count" not in cell


def test_replace_identical_source_is_an_error(nb_path):
    before = nb_path.read_bytes()
    out = run(nb_path, "replace 2\ny = 2")
    assert "did not change" in out
    assert nb_path.read_bytes() == before


def test_replace_with_empty_source_empties_cell(nb_path):
    run(nb_path, "replace 2\n")
    assert read_nb(nb_path)["cells"][2]["source"] == []


# --- insert -----------------------------------------------------------------------------


@pytest.mark.parametrize("index", [0, 1, 3])  # start, middle, end (append)
def test_insert_positions(nb_path, index):
    run(nb_path, f"insert {index}\nnew = 1")
    cells = read_nb(nb_path)["cells"]
    assert len(cells) == 4
    assert source_text(cells[index]) == "new = 1"


def test_insert_defaults_to_code_and_has_unique_id(nb_path):
    run(nb_path, "insert 1\na = 1")
    run(nb_path, "insert 1\nb = 2")
    cells = read_nb(nb_path)["cells"]
    assert cells[1]["cell_type"] == "code"
    assert cells[1]["outputs"] == [] and cells[1]["execution_count"] is None
    ids = [c["id"] for c in cells]
    assert len(set(ids)) == len(ids)


def test_insert_markdown_has_no_outputs(nb_path):
    run(nb_path, "insert 0 markdown\n## Heading")
    cell = read_nb(nb_path)["cells"][0]
    assert cell["cell_type"] == "markdown"
    assert "outputs" not in cell and "execution_count" not in cell


def test_insert_into_old_notebook_adds_no_id(tmp_path):
    path = write_nb(tmp_path / "old.ipynb", make_notebook(minor=4))
    run(path, "insert 0\nx = 1")
    assert all("id" not in c for c in read_nb(path)["cells"])


def test_insert_past_the_end_is_an_error(nb_path):
    out = run(nb_path, "insert 4\nx = 1")
    assert "out of range" in out
    assert len(read_nb(nb_path)["cells"]) == 3


# --- delete -----------------------------------------------------------------------------


def test_delete_cell(nb_path):
    out = run(nb_path, "delete 1")
    assert "Deleted code cell 1" in out and "print('hello')" in out
    cells = read_nb(nb_path)["cells"]
    assert [c["id"] for c in cells] == ["md000001", "code0002"]


def test_delete_last_remaining_cell_leaves_valid_notebook(tmp_path):
    nb = make_notebook()
    nb["cells"] = nb["cells"][:1]
    path = write_nb(tmp_path / "one.ipynb", nb)
    run(path, "delete 0")
    assert read_nb(path)["cells"] == []


# --- clear_outputs ----------------------------------------------------------------------


def test_clear_outputs_all(nb_path):
    out = run(nb_path, "clear_outputs")
    assert "Cleared outputs of 1 cell" in out
    cell = read_nb(nb_path)["cells"][1]
    assert cell["outputs"] == [] and cell["execution_count"] is None
    assert source_text(cell) == "print('hello')\nx = 1"  # source untouched


def test_clear_outputs_single_cell_and_nothing_to_clear(nb_path):
    assert "Cleared outputs of cell 1" in run(nb_path, "clear_outputs 1")
    assert "no outputs to clear" in run(nb_path, "clear_outputs 1")
    assert "no outputs to clear" in run(nb_path, "clear_outputs 0")  # markdown cell


# --- errors -----------------------------------------------------------------------------


@pytest.mark.parametrize("op", ["replace", "delete"])
@pytest.mark.parametrize("index", [3, 99, -1])
def test_index_out_of_range(nb_path, op, index):
    before = nb_path.read_bytes()
    out = run(nb_path, f"{op} {index}\nx")
    assert "out of range" in out
    assert nb_path.read_bytes() == before


def test_clear_outputs_index_out_of_range(nb_path):
    assert "out of range" in run(nb_path, "clear_outputs 7")


def test_missing_file(tmp_path):
    assert "No such file" in run(tmp_path / "nope.ipynb", "list")


def test_wrong_extension(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("{}")
    assert "not a notebook" in run(path, "list")


def test_invalid_json_is_not_overwritten(tmp_path):
    path = tmp_path / "broken.ipynb"
    path.write_text("{ not json")
    assert "not valid JSON" in run(path, "replace 0\nx")
    assert path.read_text() == "{ not json"


def test_json_without_cells(tmp_path):
    path = tmp_path / "empty.ipynb"
    path.write_text('{"nbformat": 4}')
    assert "no `cells` list" in run(path, "list")


def test_old_nbformat_rejected(tmp_path):
    nb = make_notebook()
    nb["nbformat"] = 3
    path = write_nb(tmp_path / "v3.ipynb", nb)
    assert "Unsupported notebook format" in run(path, "list")


def test_unknown_operation_and_empty_request(nb_path):
    assert "unknown operation" in run(nb_path, "frobnicate 1")
    out = "\n".join(m.content for m in execute_notebook("", [str(nb_path)], None))
    assert "No notebook operation" in out


def test_relative_path_escaping_cwd_is_rejected(tmp_path, monkeypatch):
    sub = tmp_path / "sub"
    sub.mkdir()
    write_nb(tmp_path / "outside.ipynb", make_notebook())
    monkeypatch.chdir(sub)
    out = "\n".join(
        m.content for m in execute_notebook("list", ["../outside.ipynb"], None)
    )
    assert "Path traversal" in out


def test_relative_path_inside_cwd_works(tmp_path, monkeypatch):
    write_nb(tmp_path / "here.ipynb", make_notebook())
    monkeypatch.chdir(tmp_path)
    out = "\n".join(m.content for m in execute_notebook("list", ["here.ipynb"], None))
    assert "3 cells" in out


# --- file fidelity ----------------------------------------------------------------------


def test_unicode_survives(nb_path):
    run(nb_path, "replace 2\nname = 'Zoë ✓ 日本語'")
    raw = nb_path.read_text(encoding="utf-8")
    assert "Zoë ✓ 日本語" in raw  # written as-is, not as \u escapes
    assert source_text(read_nb(nb_path)["cells"][2]) == "name = 'Zoë ✓ 日本語'"


@pytest.mark.parametrize("indent", [1, 2, 4])
def test_indent_is_preserved(tmp_path, indent):
    path = write_nb(tmp_path / "t.ipynb", make_notebook(), indent=indent)
    run(path, "replace 2\ny = 3")
    assert path.read_text().split("\n")[1] == " " * indent + '"cells": ['


def test_missing_trailing_newline_is_preserved(tmp_path):
    path = tmp_path / "t.ipynb"
    path.write_text(json.dumps(make_notebook(), indent=1))  # no trailing newline
    run(path, "replace 2\ny = 3")
    assert not path.read_text().endswith("\n")


def test_crlf_line_endings_are_preserved(tmp_path):
    path = tmp_path / "t.ipynb"
    text = json.dumps(make_notebook(), indent=1) + "\n"
    path.write_bytes(text.replace("\n", "\r\n").encode())
    run(path, "replace 2\ny = 3")
    raw = path.read_bytes()
    assert b"\r\n" in raw
    assert b"\n" not in raw.replace(b"\r\n", b"")


def test_file_permissions_are_preserved(nb_path):
    nb_path.chmod(0o640)
    run(nb_path, "replace 2\ny = 3")
    assert nb_path.stat().st_mode & 0o777 == 0o640


def test_no_temp_files_left_behind(nb_path):
    run(nb_path, "replace 2\ny = 3")
    run(nb_path, "replace 2\ny = 2")
    assert [p.name for p in nb_path.parent.iterdir()] == ["test.ipynb"]


def test_failed_write_keeps_original_and_cleans_up(nb_path, monkeypatch):
    before = nb_path.read_bytes()

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("gptme.tools.notebook.os.replace", boom)
    with pytest.raises(OSError, match="disk full"):
        list(execute_notebook_impl("replace 2\ny = 3", nb_path))
    assert nb_path.read_bytes() == before
    assert [p.name for p in nb_path.parent.iterdir()] == ["test.ipynb"]


def test_symlink_is_edited_in_place(tmp_path):
    real = write_nb(tmp_path / "real.ipynb", make_notebook())
    link = tmp_path / "link.ipynb"
    link.symlink_to(real)
    run(link, "replace 2\ny = 3")
    assert link.is_symlink()
    assert source_text(read_nb(real)["cells"][2]) == "y = 3"


def test_output_is_a_valid_notebook(nb_path):
    nbformat = pytest.importorskip("nbformat")
    run(nb_path, "insert 1 markdown\n## Added")
    run(nb_path, "replace 3\nprint('x')")
    run(nb_path, "delete 0")
    nbformat.validate(nbformat.read(str(nb_path), as_version=4))


# --- tool-call (function-calling) format -------------------------------------------------


def test_tool_format_kwargs(nb_path):
    kwargs = {
        "path": str(nb_path),
        "operation": "replace",
        "cell_index": "2",
        "source": "y = 'via kwargs'",
    }
    out = "\n".join(m.content for m in execute_notebook(None, None, kwargs))
    assert "Replaced cell 2" in out
    assert source_text(read_nb(nb_path)["cells"][2]) == "y = 'via kwargs'"


def test_tool_format_insert_with_type_and_list(nb_path):
    kwargs = {
        "path": str(nb_path),
        "operation": "insert",
        "cell_index": "0",
        "cell_type": "markdown",
        "source": "# First",
    }
    list(execute_notebook(None, None, kwargs))
    assert read_nb(nb_path)["cells"][0]["cell_type"] == "markdown"

    out = "\n".join(
        m.content
        for m in execute_notebook(
            None, None, {"path": str(nb_path), "operation": "list"}
        )
    )
    assert "4 cells" in out


# --- registration -----------------------------------------------------------------------


def test_tool_is_discovered():
    from gptme.tools import _discover_tools

    tools = {t.name: t for t in _discover_tools(["gptme.tools"])}
    assert "notebook" in tools
    assert tools["notebook"].block_types == ["notebook"]
