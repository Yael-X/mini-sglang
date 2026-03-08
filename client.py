import json, sys, urllib.request

req = urllib.request.Request("http://localhost:8000/v1/chat/completions", 
    data = json.dumps({
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "你好"}],
    "stream": True,
    "max_tokens": 512,  # 显式增加长度限制
    "temperature": 0.7  # 略微增加随机性，防止思考陷入死循环
}).encode(),
    headers={"Content-Type": "application/json"})

with urllib.request.urlopen(req) as f:
    full_text = ""
    for line in f:
        line = line.decode().strip()
        if line.startswith("data: {"):
            content = json.loads(line[6:])["choices"][0]["delta"].get("content", "")
            full_text += content
    # 打印去除思考标签后的内容
    print(full_text.split("</think>")[-1].strip())

