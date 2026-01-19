# tools.py
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Optional


ALLOWED_EXTS = {".txt", ".md", ".py", ".json", ".yaml", ".yml", ".log"}


@dataclass
class SearchHit:
    path: str
    line_no: int
    line: str


class ToolError(Exception):
    pass


def _resolve_under_root(root_dir: str, target_path: str) -> Path:
    """
    root_dir 밖으로 나가는 경로 접근(../)을 막기 위한 안전장치.
    """
    root = Path(root_dir).expanduser().resolve()
    target = Path(target_path).expanduser()

    # 상대경로면 root 기준으로 붙임
    if not target.is_absolute():
        target = (root / target).resolve()
    else:
        target = target.resolve()

    try:
        target.relative_to(root)
    except ValueError:
        raise ToolError(f"Access denied: path is outside root_dir. path={target}")

    return target


def list_files(root_dir: str, max_files: int = 200) -> List[str]:
    """
    root_dir 아래의 허용 확장자 파일 목록을 반환.
    """
    root = Path(root_dir).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ToolError(f"root_dir not found or not a directory: {root}")

    results: List[str] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTS:
            results.append(str(p))
            if len(results) >= max_files:
                break

    return results


def read_file(root_dir: str, path: str, max_chars: int = 12_000) -> str:
    """
    파일을 읽되 너무 길면 앞부분만 잘라서 반환.
    """
    p = _resolve_under_root(root_dir, path)

    if p.suffix.lower() not in ALLOWED_EXTS:
        raise ToolError(f"File extension not allowed: {p.suffix}")

    if not p.exists() or not p.is_file():
        raise ToolError(f"File not found: {p}")

    text = p.read_text(encoding="utf-8", errors="replace")
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n...[TRUNCATED]..."
    return text


def search_files(
    root_dir: str,
    query: str,
    max_hits: int = 25,
    context_lines: int = 1,
    case_sensitive: bool = False,
) -> List[SearchHit]:
    """
    root_dir 내 파일들에서 query 문자열(또는 정규식)을 검색.
    - 기본은 '문자열 포함' 검색
    - query가 /.../ 형태면 정규식으로 처리
    """
    root = Path(root_dir).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ToolError(f"root_dir not found or not a directory: {root}")

    # 정규식 모드 지원: /pattern/
    regex_mode = False
    pattern = query
    if len(query) >= 2 and query.startswith("/") and query.endswith("/"):
        regex_mode = True
        pattern = query[1:-1]

    flags = 0 if case_sensitive else re.IGNORECASE
    compiled: Optional[re.Pattern] = None
    if regex_mode:
        try:
            compiled = re.compile(pattern, flags)
        except re.error as e:
            raise ToolError(f"Invalid regex: {e}")

    hits: List[SearchHit] = []

    for p in root.rglob("*"):
        if not (p.is_file() and p.suffix.lower() in ALLOWED_EXTS):
            continue

        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue

        for i, line in enumerate(lines, start=1):
            matched = False
            if regex_mode:
                assert compiled is not None
                matched = compiled.search(line) is not None
            else:
                if case_sensitive:
                    matched = query in line
                else:
                    matched = query.lower() in line.lower()

            if matched:
                # 주변 문맥 붙여주기(간단)
                start = max(1, i - context_lines)
                end = min(len(lines), i + context_lines)
                snippet = []
                for ln in range(start, end + 1):
                    prefix = ">" if ln == i else " "
                    snippet.append(f"{prefix}{ln:>4}: {lines[ln-1]}")
                joined = "\n".join(snippet)

                hits.append(SearchHit(path=str(p), line_no=i, line=joined))
                if len(hits) >= max_hits:
                    return hits

    return hits
