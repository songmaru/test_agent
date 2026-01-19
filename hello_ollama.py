import requests

payload = {
    "model": "qwen3:8b",
    "messages": [
        {"role": "user", "content": "한 문장으로 오늘 할 일을 정리해줘."}
    ],
    "stream": False
}

r = requests.post("http://localhost:11434/api/chat", json=payload, timeout=120)
r.raise_for_status()
print(r.json()["message"]["content"])
