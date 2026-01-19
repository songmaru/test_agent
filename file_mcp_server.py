# file_mcp_server.py
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Dict, Any

from mcp.server.fastmcp import FastMCP

# -----------------------
# 설정: 여기만 바꾸면 됨
# -----------------------
ROOT_DIR = os.path.abspath("data")
ALLOWED_EXTS = {".txt", ".md", ".py", ".json", ".yaml", ".yml", ".log"}

mcp = FastMCP("LocalFileTools")


class ToolError(Exception):
    pass


def _resolve_under_root(path: str) -> Path:
    root = Path(ROOT_DIR).resolve()
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = (root / p).resolve()
    else:
        p = p.resolve()

    # root 밖으로 나가면 차단
    try:
        p.relative_to(root)
    except ValueError:
        raise ToolError(f"Access denied (outside ROOT_DIR): {p}")
    return p


def _iter_files(max_files: int = 500):
    root = Path(ROOT_DIR).resolve()
    count = 0
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTS:
            yield p
            count += 1
            if count >= max_files:
                return


@mcp.tool()
def list_files(max_files: int = 200) -> List[str]:
    """List files under ROOT_DIR (allowed extensions only)."""
    return [str(p) for p in _iter_files(max_files=max_files)]


@mcp.tool()
def read_file(path: str, max_chars: int = 12000) -> str:
    """Read a file safely under ROOT_DIR (returns truncated text if large)."""
    p = _resolve_under_root(path)
    if p.suffix.lower() not in ALLOWED_EXTS:
        raise ToolError(f"File extension not allowed: {p.suffix}")
    if not p.exists() or not p.is_file():
        raise ToolError(f"File not found: {p}")

    text = p.read_text(encoding="utf-8", errors="replace")
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n...[TRUNCATED]..."
    return text


@mcp.tool()
def search_files(
    query: str,
    max_hits: int = 25,
    context_lines: int = 1,
    case_sensitive: bool = False,
) -> List[Dict[str, Any]]:
    """
    Search for a substring (or regex if query is /pattern/) in files under ROOT_DIR.
    Returns hits with file path and line snippets.
    """
    if not query or not query.strip():
        raise ToolError("query must be non-empty")

    regex_mode = len(query) >= 2 and query.startswith("/") and query.endswith("/")
    pattern = query[1:-1] if regex_mode else query

    flags = 0 if case_sensitive else re.IGNORECASE
    compiled = re.compile(pattern, flags) if regex_mode else None

    hits: List[Dict[str, Any]] = []

    for p in _iter_files(max_files=1000):
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue

        for i, line in enumerate(lines, start=1):
            matched = False
            if regex_mode:
                matched = compiled.search(line) is not None  # type: ignore
            else:
                matched = (pattern in line) if case_sensitive else (pattern.lower() in line.lower())

            if matched:
                start = max(1, i - context_lines)
                end = min(len(lines), i + context_lines)
                snippet = []
                for ln in range(start, end + 1):
                    prefix = ">" if ln == i else " "
                    snippet.append(f"{prefix}{ln:>4}: {lines[ln-1]}")
                hits.append(
                    {
                        "path": str(p),
                        "line_no": i,
                        "snippet": "\n".join(snippet),
                    }
                )
                if len(hits) >= max_hits:
                    return hits

    return hits


if __name__ == "__main__":
    # stdio로 실행 (로컬 연결 표준)
    mcp.run(transport="stdio")
