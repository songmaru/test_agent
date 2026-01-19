# mcp_agent.py
from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, List

import requests
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:8b"


def ollama_chat(messages: List[Dict[str, Any]]) -> str:
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.2},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=180)
    r.raise_for_status()
    return r.json()["message"]["content"]


def parse_json_action(s: str) -> Dict[str, Any]:
    s = s.strip()
    if not s.startswith("{"):
        a = s.find("{")
        b = s.rfind("}")
        if a != -1 and b != -1 and b > a:
            s = s[a : b + 1]
    return json.loads(s)


def calltoolresult_to_text(result: types.CallToolResult) -> str:
    # MCP tool 결과는 content 배열로 들어옴 (TextContent 등)
    parts = []
    for c in result.content:
        if isinstance(c, types.TextContent):
            parts.append(c.text)
        else:
            # 모르는 타입은 JSON으로 덤프
            parts.append(c.model_dump_json())
    return "\n".join(parts).strip()


def mcp_tools_to_prompt(tools: List[types.Tool]) -> str:
    # LLM에게 보여줄 “도구 설명” 문자열
    lines = []
    for t in tools:
        schema = t.inputSchema if hasattr(t, "inputSchema") else None
        lines.append(f"- {t.name}: {t.description or ''}")
        if schema:
            # 너무 길어지지 않게 최소로만
            lines.append(f"  inputSchema keys: {list(schema.get('properties', {}).keys())}")
    return "\n".join(lines)


async def run_agent():
    # MCP 서버를 subprocess로 띄워 붙는 방식(stdio)
    server_params = StdioServerParameters(
        command="python",
        args=["file_mcp_server.py"],
        env=os.environ.copy(),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()  # MCP 핸드셰이크
            tools_resp = await session.list_tools()
            tools = tools_resp.tools

            tools_text = mcp_tools_to_prompt(tools)

            system = f"""
You are a local file agent.
You can use MCP tools via the client.

You MUST output ONLY valid JSON.
Choose exactly one:
- Call a tool:
  {{ "tool": "<tool_name>", "args": {{...}} }}
- Final answer:
  {{ "tool": "final", "args": {{ "answer": "...", "citations": ["..."] }} }}

Available tools:
{tools_text}

Rules:
1) Prefer search_files first when you need file info.
2) Only read_file for the most relevant files.
3) Never invent file content. Use tool results only.
4) citations can be:
   - file:<path>#L<line_no> (from search_files)
   - file:<path>#(read) (from read_file)
""".strip()

            print("MCP connected. Type a question. Ctrl+C to exit.\n")

            while True:
                q = input("Q> ").strip()
                if not q:
                    continue

                messages: List[Dict[str, Any]] = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": q},
                ]

                max_steps = 6
                for _ in range(max_steps):
                    raw = ollama_chat(messages)
                    try:
                        action = parse_json_action(raw)
                    except Exception as e:
                        messages.append(
                            {"role": "user", "content": f"Invalid JSON ({e}). Output ONLY JSON."}
                        )
                        continue

                    tool = action.get("tool")
                    args = action.get("args", {})

                    if tool == "final":
                        print("\n--- ANSWER ---")
                        print(action["args"]["answer"])
                        if action["args"].get("citations"):
                            print("\n--- CITATIONS ---")
                            for c in action["args"]["citations"]:
                                print("-", c)
                        print("--------------\n")
                        break

                    # MCP tool 호출
                    try:
                        result = await session.call_tool(tool, args)
                        obs_text = calltoolresult_to_text(result)
                    except Exception as e:
                        obs_text = f"TOOL_ERROR: {type(e).__name__}: {e}"

                    # 다음 턴에 관찰 결과를 넣어줌
                    messages.append({"role": "assistant", "content": json.dumps(action, ensure_ascii=False)})
                    messages.append({"role": "user", "content": "Observation:\n" + obs_text})
                else:
                    print("Max steps reached. Try a narrower question.\n")


if __name__ == "__main__":
    asyncio.run(run_agent())
