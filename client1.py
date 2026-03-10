import json, sys, urllib.request

# 从命令行获取问题，如果没有输入则默认问“你好”
query = sys.argv[1] if len(sys.argv) > 1 else "你好"

url = "http://localhost:8000/v1/chat/completions"
# 调高 max_tokens 以防止推理中途被掐断
data = json.dumps({
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": query}],
    "stream": True,
    "max_tokens": 2048,
    "temperature": 0.6
}).encode()

req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

print(f"--- 用户问题: {query} ---")
try:
    with urllib.request.urlopen(req) as f:
        full_text = ""
        for line in f:
            line = line.decode().strip()
            if line.startswith("data: {"):
                try:
                    content = json.loads(line[6:])["choices"][0]["delta"].get("content", "")
                    full_text += content
                    # 实时打印，方便观察它是否在“思考”
                    print(content, end="", flush=True)
                except (KeyError, IndexError):
                    continue
        
        # 换行后显示过滤掉思考过程的最终答案
        if "</think>" in full_text:
            final_answer = full_text.split("</think>")[-1].strip()
            print(f"\n\n--- 最终回答 ---\n{final_answer}")
        else:
            print("\n\n(未检测到完整思考闭合标签)")
except Exception as e:
    print(f"\n请求出错: {e}")

