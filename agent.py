# agent.py
from __future__ import annotations

import json
import os
import textwrap
from typing import Any, Dict, List

import requests

from tools import list_files, read_file, search_files, ToolError, SearchHit


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:8b"

# 안전: 에이전트가 접근 가능한 루트 폴더 (원하는 경로로 바꾸세요)
ROOT_DIR = os.path.abspath("data")

# 에이전트가 호출 가능한 도구 목록 (allowlist)
ALLOWED_TOOLS = {"list_files", "search_files", "read_file", "final"}


SYSTEM_PROMPT = f"""
You are a careful file assistant running locally.

You have access to tools for working with files UNDER this root directory ONLY:
root_dir = {ROOT_DIR}

You MUST respond with ONLY valid JSON (no markdown, no extra text).
Choose exactly one of:
- tool call:  {{ "tool": "list_files|search_files|read_file", "args": {{...}} }}
- final:      {{ "tool": "final", "args": {{ "answer": "...", "citations": ["..."] }} }}

Rules:
1) If you need information from files, use search_files first.
2) Use read_file only for the most relevant files, and keep it minimal.
3) Never invent file contents. Base answers on tool observations only.
4) citations should be simple strings like:
   - "file:<path>#L<start>-L<end>" when you cite line ranges from search snippets
   - or "file:<path>#(read)" when you cite full/partial read.
"""


def ollama_chat(messages: List[Dict[str, Any]]) -> str:
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        # 옵션: 너무 길게 말하지 않게
        "options": {"temperature": 0.2},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=180)
    r.raise_for_status()
    return r.json()["message"]["content"]


def parse_json_strict(s: str) -> Dict[str, Any]:
    """
    모델이 가끔 앞뒤로 텍스트를 섞을 수 있어 방어적으로 JSON만 추출.
    그래도 실패하면 그대로 에러.
    """
    s = s.strip()
    # JSON 객체 시작/끝을 찾아보기
    if not s.startswith("{"):
        first = s.find("{")
        last = s.rfind("}")
        if first != -1 and last != -1 and last > first:
            s = s[first : last + 1]

    return json.loads(s)


def tool_to_observation(result: Any) -> str:
    """
    tool 실행 결과를 LLM에 '관찰(observation)'로 전달하기 위한 문자열 변환.
    """
    if isinstance(result, list):
        # SearchHit list인 경우 보기 좋게
        if result and isinstance(result[0], SearchHit):
            lines = []
            for idx, hit in enumerate(result, start=1):
                lines.append(f"[HIT {idx}] path={hit.path}")
                lines.append(hit.line)
                lines.append("")
            return "\n".join(lines).strip()

        # 일반 리스트
        return json.dumps(result, ensure_ascii=False, indent=2)

    if isinstance(result, str):
        return result

    return json.dumps(result, ensure_ascii=False, indent=2)


def run_tool(tool: str, args: Dict[str, Any]) -> Any:
    if tool not in ALLOWED_TOOLS:
        raise ToolError(f"Tool not allowed: {tool}")

    if tool == "list_files":
        return list_files(ROOT_DIR, max_files=int(args.get("max_files", 200)))

    if tool == "search_files":
        query = str(args.get("query", "")).strip()
        if not query:
            raise ToolError("search_files requires non-empty 'query'")
        return search_files(
            ROOT_DIR,
            query=query,
            max_hits=int(args.get("max_hits", 25)),
            context_lines=int(args.get("context_lines", 1)),
            case_sensitive=bool(args.get("case_sensitive", False)),
        )

    if tool == "read_file":
        path = str(args.get("path", "")).strip()
        if not path:
            raise ToolError("read_file requires 'path'")
        return read_file(
            ROOT_DIR,
            path=path,
            max_chars=int(args.get("max_chars", 12000)),
        )

    raise ToolError(f"Unknown tool: {tool}")


def agent(question: str, max_steps: int = 6) -> Dict[str, Any]:
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT.strip()},
        {"role": "user", "content": question},
    ]

    for step in range(1, max_steps + 1):
        raw = ollama_chat(messages)

        try:
            action = parse_json_strict(raw)
        except Exception as e:
            # 모델이 JSON 규칙을 어겼을 때: 규칙을 재강조하고 재시도
            messages.append(
                {
                    "role": "assistant",
                    "content": '{"tool":"search_files","args":{"query":"(rule reminder) output JSON only"}}',
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": f"Your last output was not valid JSON. Error={e}. Output ONLY JSON per the schema.",
                }
            )
            continue

        tool = action.get("tool")
        args = action.get("args", {})

        if tool == "final":
            return action

        # tool 실행
        try:
            result = run_tool(tool, args)
            obs = tool_to_observation(result)
        except Exception as e:
            obs = f"TOOL_ERROR: {type(e).__name__}: {e}"

        # 관찰 결과를 메시지에 추가
        messages.append({"role": "assistant", "content": json.dumps(action, ensure_ascii=False)})
        messages.append(
            {
                "role": "user",
                "content": "Observation (tool result):\n" + obs,
            }
        )

    # max_steps 초과
    return {
        "tool": "final",
        "args": {
            "answer": "최대 단계에 도달해서 작업을 중단했어. 질문을 더 구체화하거나, 검색 키워드를 좁혀줘.",
            "citations": [],
        },
    }


def main():
    print(f"ROOT_DIR = {ROOT_DIR}")
    print("Type a question. Ctrl+C to exit.\n")

    while True:
        q = input("Q> ").strip()
        if not q:
            continue
        out = agent(q, max_steps=6)
        print("\n--- ANSWER ---")
        print(out["args"]["answer"])
        if out["args"].get("citations"):
            print("\n--- CITATIONS ---")
            for c in out["args"]["citations"]:
                print("-", c)
        print("--------------\n")


if __name__ == "__main__":
    main()
