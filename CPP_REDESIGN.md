# Mini-SGLang C++ 重写设计文档

## 项目分析

### 项目概述
Mini-SGLang 是一个轻量级、高性能的大语言模型（LLM）推理框架，当前使用 Python 实现，包含约 5,000 行核心代码。主要特点：

1. **分布式架构**：多进程设计，支持 Tensor Parallelism (TP)
2. **高性能优化**：
   - Radix Cache（共享前缀 KV 缓存复用）
   - Chunked Prefill（分块预填充）
   - Overlap Scheduling（重叠调度）
   - CUDA Graph 优化
   - FlashAttention/FlashInfer 集成

3. **核心组件**：
   - API Server（FastAPI）
   - Tokenizer/Detokenizer Worker
   - Scheduler Worker（每个 GPU 一个）
   - Engine（推理引擎）

### 当前 Python 架构分析

#### 1. 核心模块 (`minisgl.core`)
- **Req**: 请求数据结构，包含 input_ids、采样参数、缓存句柄等
- **Batch**: 批次数据结构，管理多个请求
- **Context**: 全局上下文，管理当前批次和注意力后端
- **SamplingParams**: 采样参数（temperature, top_k, top_p 等）

#### 2. 引擎模块 (`minisgl.engine`)
- **Engine**: 核心推理引擎
  - 模型加载和权重管理
  - KV Cache 分配
  - CUDA Graph 管理
  - 前向传播和采样

#### 3. 调度器模块 (`minisgl.scheduler`)
- **Scheduler**: 请求调度器
  - PrefillManager: 预填充管理（支持分块）
  - DecodeManager: 解码管理
  - CacheManager: 缓存管理（Naive/Radix）
  - TableManager: 页表管理
  - I/O 处理（ZeroMQ 通信）

#### 4. 模型模块 (`minisgl.models`)
- **BaseLLMModel**: 模型基类
- **LlamaModel**: Llama 系列实现
- **Qwen3Model**: Qwen3 系列实现
- 权重加载和分片

#### 5. 层模块 (`minisgl.layers`)
- Linear, LayerNorm, Embedding, RoPE, Activation
- 支持 Tensor Parallelism

#### 6. 注意力模块 (`minisgl.attention`)
- **BaseAttnBackend**: 注意力后端接口
- **FlashAttentionBackend**: FlashAttention 实现
- **FlashInferBackend**: FlashInfer 实现

#### 7. KV Cache 模块 (`minisgl.kvcache`)
- **BaseKVCache**: KV 缓存接口
- **MHAKVCache**: 多头注意力 KV 缓存
- **BaseCacheManager**: 缓存管理器接口
- **NaiveCacheManager**: 简单缓存管理
- **RadixCacheManager**: Radix 树缓存管理

#### 8. 内核模块 (`minisgl.kernel`)
- CUDA 内核实现（C++/CUDA）
- Tensor 操作
- NCCL 通信封装
- Radix 树操作

#### 9. 分布式模块 (`minisgl.distributed`)
- Tensor Parallelism 支持
- NCCL/Gloo 通信
- 分布式信息管理

#### 10. 消息模块 (`minisgl.message`)
- ZeroMQ 消息序列化/反序列化
- 前后端消息定义

#### 11. 服务器模块 (`minisgl.server`)
- FastAPI HTTP 服务器
- OpenAI 兼容 API
- 进程启动管理

#### 12. Tokenizer 模块 (`minisgl.tokenizer`)
- Tokenization/Detokenization
- 独立 Worker 进程

### 技术依赖

**Python 依赖**:
- PyTorch（核心计算框架）
- transformers（模型加载）
- flashinfer-python（注意力后端）
- sgl-kernel（CUDA 内核）
- pyzmq（进程间通信）
- fastapi/uvicorn（HTTP 服务器）
- msgpack（序列化）

**系统依赖**:
- CUDA Toolkit
- NCCL（多 GPU 通信）
- ZeroMQ（进程间通信）

## C++ 重写目录结构设计

### 推荐目录结构

```
mini-sglang-cpp/
├── CMakeLists.txt                 # 主 CMake 配置文件
├── README.md
├── LICENSE
├── .gitignore
│
├── include/                       # 公共头文件
│   └── minisgl/
│       ├── core/                  # 核心数据结构
│       │   ├── req.hpp            # 请求结构
│       │   ├── batch.hpp             # 批次结构
│       │   ├── context.hpp          # 上下文
│       │   └── sampling_params.hpp  # 采样参数
│       │
│       ├── engine/                 # 引擎接口
│       │   ├── engine.hpp
│       │   └── engine_config.hpp
│       │
│       ├── scheduler/              # 调度器接口
│       │   ├── scheduler.hpp
│       │   └── scheduler_config.hpp
│       │
│       ├── models/                 # 模型接口
│       │   ├── base_model.hpp
│       │   ├── llama_model.hpp
│       │   └── qwen3_model.hpp
│       │
│       ├── layers/                 # 神经网络层接口
│       │   ├── base_layer.hpp
│       │   ├── linear.hpp
│       │   ├── norm.hpp
│       │   ├── embedding.hpp
│       │   ├── rotary.hpp
│       │   └── activation.hpp
│       │
│       ├── attention/              # 注意力后端接口
│       │   ├── base_backend.hpp
│       │   ├── flash_attention.hpp
│       │   └── flash_infer.hpp
│       │
│       ├── kvcache/                # KV 缓存接口
│       │   ├── base_cache.hpp
│       │   ├── cache_manager.hpp
│       │   └── radix_tree.hpp
│       │
│       ├── distributed/            # 分布式通信接口
│       │   ├── nccl_comm.hpp
│       │   └── distributed_info.hpp
│       │
│       ├── message/                # 消息定义
│       │   ├── message.hpp
│       │   └── serializer.hpp
│       │
│       ├── utils/                  # 工具函数
│       │   ├── logger.hpp
│       │   ├── tensor_utils.hpp
│       │   └── cuda_utils.hpp
│       │
│       └── kernel/                 # CUDA 内核接口
│           ├── tensor.hpp
│           └── radix.hpp
│
├── src/                            # 实现文件
│   ├── core/                       # 核心数据结构实现
│   │   ├── req.cpp
│   │   ├── batch.cpp
│   │   └── context.cpp
│   │
│   ├── engine/                     # 引擎实现
│   │   ├── engine.cpp
│   │   ├── graph_runner.cpp        # CUDA Graph 管理
│   │   └── sampler.cpp             # 采样器
│   │
│   ├── scheduler/                  # 调度器实现
│   │   ├── scheduler.cpp
│   │   ├── prefill_manager.cpp
│   │   ├── decode_manager.cpp
│   │   ├── cache_manager.cpp
│   │   ├── table_manager.cpp
│   │   └── io_handler.cpp          # ZeroMQ I/O
│   │
│   ├── models/                     # 模型实现
│   │   ├── base_model.cpp
│   │   ├── llama_model.cpp
│   │   └── qwen3_model.cpp
│   │
│   ├── layers/                     # 层实现
│   │   ├── linear.cpp
│   │   ├── norm.cpp
│   │   ├── embedding.cpp
│   │   ├── rotary.cpp
│   │   └── activation.cpp
│   │
│   ├── attention/                  # 注意力后端实现
│   │   ├── flash_attention.cpp
│   │   └── flash_infer.cpp
│   │
│   ├── kvcache/                    # KV 缓存实现
│   │   ├── mha_cache.cpp
│   │   ├── naive_manager.cpp
│   │   └── radix_manager.cpp
│   │
│   ├── distributed/                # 分布式通信实现
│   │   ├── nccl_comm.cpp
│   │   └── pynccl.cpp              # PyNCCL 替代
│   │
│   ├── message/                    # 消息序列化实现
│   │   ├── message.cpp
│   │   └── serializer.cpp
│   │
│   ├── server/                     # HTTP 服务器
│   │   ├── api_server.cpp          # 使用 cpp-httplib 或类似
│   │   └── launch.cpp                # 进程启动
│   │
│   ├── tokenizer/                  # Tokenizer Worker
│   │   ├── tokenizer_worker.cpp
│   │   └── hf_tokenizer.cpp        # HuggingFace tokenizer 绑定
│   │
│   └── utils/                      # 工具实现
│       ├── logger.cpp
│       ├── tensor_utils.cpp
│       └── cuda_utils.cpp
│
├── kernel/                         # CUDA 内核代码
│   ├── include/
│   │   └── minisgl/
│   │       ├── tensor.h
│   │       ├── utils.h
│   │       ├── utils.cuh
│   │       ├── warp.cuh
│   │       └── nccl227.h
│   │
│   ├── src/
│   │   ├── tensor.cu
│   │   ├── radix.cu
│   │   └── pynccl.cu
│   │
│   └── jit/                        # JIT 编译的内核
│       ├── index.cu
│       └── store.cu
│
├── third_party/                    # 第三方依赖（可选，使用 git submodule）
│   ├── json/                       # JSON 库（nlohmann/json）
│   ├── httplib/                    # HTTP 服务器（可选）
│   └── spdlog/                     # 日志库（可选）
│
├── tests/                          # 测试代码
│   ├── unit/                       # 单元测试
│   │   ├── test_core.cpp
│   │   ├── test_scheduler.cpp
│   │   ├── test_kvcache.cpp
│   │   └── test_tensor.cpp
│   │
│   ├── integration/                # 集成测试
│   │   └── test_api.cpp
│   │
│   └── benchmark/                  # 性能测试
│       ├── offline_bench.cpp
│       └── online_bench.cpp
│
├── examples/                       # 示例代码
│   ├── simple_inference.cpp
│   └── api_server_example.cpp
│
├── scripts/                        # 构建和部署脚本
│   ├── build.sh
│   ├── setup_env.sh
│   └── run_tests.sh
│
└── docs/                           # 文档
    ├── architecture.md
    ├── api_reference.md
    └── build_guide.md
```

## 关键技术选型建议

### 1. 深度学习框架
**选项 A: LibTorch (PyTorch C++)**
- ✅ 与 Python 版本兼容性好，权重可直接复用
- ✅ API 与 PyTorch 相似，迁移成本低
- ✅ 支持 CUDA、自动微分、Tensor Parallelism
- ❌ 库体积较大

**选项 B: TensorRT-LLM / FasterTransformer**
- ✅ 性能优化更好
- ❌ 需要额外转换步骤
- ❌ 灵活性较低

**推荐**: **LibTorch**，便于迁移和兼容性

### 2. HTTP 服务器
**选项 A: cpp-httplib**
- ✅ 轻量级，单头文件
- ✅ 简单易用

**选项 B: Crow**
- ✅ 现代 C++，类似 Flask

**选项 C: Pistache**
- ✅ 高性能异步

**推荐**: **cpp-httplib** 或 **Crow**

### 3. 进程间通信
**选项 A: ZeroMQ (libzmq)**
- ✅ 与 Python 版本一致
- ✅ 成熟稳定

**选项 B: gRPC**
- ✅ 跨语言支持好
- ❌ 开销稍大

**推荐**: **ZeroMQ**，保持一致性

### 4. 序列化
**选项 A: msgpack-c**
- ✅ 与 Python 版本一致
- ✅ 高效二进制格式

**选项 B: Protocol Buffers**
- ✅ 类型安全
- ❌ 需要定义 schema

**推荐**: **msgpack-c**

### 5. JSON 处理
**推荐**: **nlohmann/json**（单头文件，现代 C++）

### 6. 日志
**推荐**: **spdlog**（高性能，现代 C++）

### 7. 配置管理
**推荐**: **yaml-cpp** 或 **toml11**（TOML 解析）

### 8. CUDA 内核
- 保持现有的 CUDA 内核代码
- 使用 CUDA Runtime API 或考虑 cuTensor

### 9. Tokenizer
**选项 A: sentencepiece (C++)**
- ✅ 支持大多数模型

**选项 B: HuggingFace tokenizers (C++)**
- ✅ 与 Python 版本完全兼容
- ✅ 支持更多模型

**推荐**: **HuggingFace tokenizers C++ 库**

### 10. 构建系统
**推荐**: **CMake**（标准、跨平台）

## 实现策略

### 阶段 1: 核心基础设施
1. 设置 CMake 构建系统
2. 实现核心数据结构（Req, Batch, Context）
3. 实现基础工具（Logger, Tensor Utils）
4. 集成 LibTorch

### 阶段 2: 模型和层
1. 实现基础层（Linear, Norm, Embedding）
2. 实现模型架构（Llama, Qwen3）
3. 实现权重加载

### 阶段 3: 引擎和调度
1. 实现 Engine
2. 实现 Scheduler
3. 实现 KV Cache 管理
4. 实现 CUDA Graph

### 阶段 4: 分布式和通信
1. 实现 NCCL 通信
2. 实现 ZeroMQ 消息传递
3. 实现 Tensor Parallelism

### 阶段 5: 服务器和 API
1. 实现 HTTP 服务器
2. 实现 Tokenizer Worker
3. 实现进程启动管理
4. 实现 OpenAI 兼容 API

### 阶段 6: 优化和测试
1. 性能优化
2. 单元测试和集成测试
3. 文档完善

## 关键设计考虑

### 1. 内存管理
- 使用智能指针（`std::shared_ptr`, `std::unique_ptr`）
- CUDA 内存使用 `c10::Tensor` 或 `at::Tensor`
- 注意避免内存泄漏和悬空指针

### 2. 线程安全
- Scheduler 需要线程安全设计
- 使用 `std::mutex`, `std::atomic` 等
- CUDA Stream 管理

### 3. 错误处理
- 使用异常或错误码
- CUDA 错误检查
- 资源清理（RAII）

### 4. 性能优化
- 避免不必要的拷贝
- 使用移动语义（`std::move`）
- 内存池管理
- CUDA Kernel 优化

### 5. 可扩展性
- 使用接口/抽象基类
- 插件化设计（注意力后端、缓存管理器等）
- 配置驱动

## 迁移注意事项

1. **Python 动态特性**: C++ 需要显式类型和模板
2. **GIL 不存在**: 需要显式线程管理
3. **内存管理**: 手动管理 CUDA 内存
4. **异常安全**: 确保资源正确释放
5. **编译时间**: 考虑使用 PIMPL 模式减少编译依赖

## 总结

C++ 重写将带来：
- ✅ 更好的性能（减少 Python 开销）
- ✅ 更低的内存占用
- ✅ 更好的部署灵活性
- ❌ 更长的开发时间
- ❌ 更复杂的构建和依赖管理

建议采用渐进式迁移策略，先实现核心功能，再逐步完善。
