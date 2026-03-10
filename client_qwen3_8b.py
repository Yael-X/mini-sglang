#!/usr/bin/env python3
"""
Qwen3-8B-FP8 推理测试脚本

用法:
    python client_qwen3_8b.py [问题]

示例:
    python client_qwen3_8b.py "你好，请介绍一下你自己"
    python client_qwen3_8b.py "写一个快速排序算法"

注意:
    需要先启动推理服务:
    python -m minisgl --model-path "Qwen/Qwen3-8B-FP8" --fp8-keep-quantized --attention-backend fi --port 1919
"""

import json
import sys
import urllib.request
import time

# 配置
SERVER_URL = "http://localhost:1919/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen3-8B-FP8"
MAX_TOKENS = 2048
TEMPERATURE = 0.7

def main():
    # 从命令行获取问题
    query = sys.argv[1] if len(sys.argv) > 1 else "你好，请介绍一下你自己"

    print(f"=== Qwen3-8B-FP8 推理测试 ===")
    print(f"服务器: {SERVER_URL}")
    print(f"问题: {query}")
    print("-" * 50)

    # 构建请求
    data = json.dumps({
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": query}],
        "stream": True,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE
    }).encode()

    req = urllib.request.Request(
        SERVER_URL,
        data=data,
        headers={"Content-Type": "application/json"}
    )

    start_time = time.time()
    first_token_time = None
    token_count = 0
    full_text = ""

    try:
        with urllib.request.urlopen(req, timeout=120) as f:
            for line in f:
                line = line.decode().strip()
                if line.startswith("data: {"):
                    try:
                        chunk = json.loads(line[6:])
                        delta = chunk["choices"][0]["delta"]
                        content = delta.get("content", "")

                        if content:
                            if first_token_time is None:
                                first_token_time = time.time()
                            token_count += 1
                            full_text += content
                            print(content, end="", flush=True)
                    except (KeyError, IndexError, json.JSONDecodeError):
                        continue
                elif line == "data: [DONE]":
                    break

        end_time = time.time()
        total_time = end_time - start_time

        # 性能统计
        print("\n" + "=" * 50)
        print(f"总耗时: {total_time:.2f}s")
        if first_token_time:
            ttft = first_token_time - start_time
            print(f"首token延迟 (TTFT): {ttft:.2f}s")
        if token_count > 0 and total_time > 0:
            tps = token_count / total_time
            print(f"生成tokens: {token_count}")
            print(f"吞吐量: {tps:.2f} tokens/s")

        # 提取最终回答（过滤思考过程）
        if "```" in full_text:
            final_answer = full_text.split("```")[-1].strip()
            print(f"\n--- 最终回答 ---\n{final_answer}")

    except urllib.error.URLError as e:
        print(f"\n连接错误: {e}")
        print("请确认推理服务已启动:")
        print("  python -m minisgl --model-path 'Qwen/Qwen3-8B-FP8' --fp8-keep-quantized --attention-backend fi --port 1919")
    except TimeoutError:
        print("\n请求超时")
    except Exception as e:
        print(f"\n请求出错: {e}")


if __name__ == "__main__":
    main()