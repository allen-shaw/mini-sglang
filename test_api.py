#!/usr/bin/env python3
"""
测试 Mini-SGLang API 服务器的示例脚本
"""

import requests
import json

# 服务器地址（根据你的实际端口调整）
BASE_URL = "http://127.0.0.1:1919"

def test_health():
    """测试服务器健康状态"""
    print("1. 检查服务器状态...")
    response = requests.get(f"{BASE_URL}/v1")
    print(f"   状态: {response.json()}")
    print()

def list_models():
    """获取可用模型列表"""
    print("2. 获取可用模型列表...")
    response = requests.get(f"{BASE_URL}/v1/models")
    models = response.json()
    print(f"   模型: {json.dumps(models, indent=2, ensure_ascii=False)}")
    print()

def chat_completion():
    """发送聊天完成请求（OpenAI 兼容格式）"""
    print("3. 发送聊天请求...")
    response = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": "test",
            "messages": [
                {"role": "user", "content": "你好，请用一句话介绍你自己"}
            ],
            "max_tokens": 100,
            "temperature": 0.7,
            "stream": False  # 设置为 True 可以启用流式输出
        },
        stream=False
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"   响应: {json.dumps(result, indent=2, ensure_ascii=False)}")
    else:
        print(f"   错误: {response.status_code} - {response.text}")
    print()

def simple_generate():
    """使用简单的 /generate 端点"""
    print("4. 使用 /generate 端点...")
    response = requests.post(
        f"{BASE_URL}/generate",
        json={
            "prompt": "The capital of France is",
            "max_tokens": 20
        },
        stream=True
    )
    
    if response.status_code == 200:
        print("   流式响应:")
        for line in response.iter_lines():
            if line:
                decoded = line.decode('utf-8')
                if decoded.startswith("data: "):
                    content = decoded[6:]
                    if content == "[DONE]":
                        break
                    print(f"   {content}", end="", flush=True)
        print()
    else:
        print(f"   错误: {response.status_code} - {response.text}")
    print()

if __name__ == "__main__":
    try:
        test_health()
        list_models()
        chat_completion()
        simple_generate()
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到服务器！")
        print("   请确保服务器正在运行在 http://127.0.0.1:1919")
        print("   如果使用了不同的端口，请修改 BASE_URL 变量")
    except Exception as e:
        print(f"❌ 发生错误: {e}")
