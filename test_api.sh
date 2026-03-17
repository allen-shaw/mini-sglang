#!/bin/bash

# 测试 API 服务器是否运行
echo "1. 检查服务器状态："
curl http://127.0.0.1:1919/v1

echo -e "\n\n2. 获取可用模型列表："
curl http://127.0.0.1:1919/v1/models

echo -e "\n\n3. 发送聊天请求（OpenAI 兼容格式）："
curl -X POST http://127.0.0.1:1919/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "test",
    "messages": [
      {"role": "user", "content": "你好，请介绍一下你自己"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'

echo -e "\n\n4. 使用简单的 /generate 端点："
curl -X POST http://127.0.0.1:1919/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello, how are you?",
    "max_tokens": 50
  }'
