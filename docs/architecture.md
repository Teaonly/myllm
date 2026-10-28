# 架构：前后端分离，stdio 交互（CGI 风格）

```
客户端 ──HTTP──> HTTP API 进程（前端） ──stdin/stdout, fd 0/1──> GPU 推理进程（后端）
```

## 职责

- **前端（HTTP API）**：面向客户端的 HTTP 服务，负责请求解析、tokenize / detokenize、会话与并发管理、流式输出（SSE / chunked）；不碰 GPU。
- **后端（GPU 推理进程）**：加载模型，专注 prefill/decode 计算；不做网络 IO。**不用 Python 开发**，仓库内所有 Python 代码只存在于 frontend/；协议（[protocol.md](protocol.md)）是两侧唯一的共享契约，各自语言实现编解码。
- **交互**：前端把请求写进后端的 stdin，后端逐 token 写回 stdout。两个独立进程，边界即模块边界。

## 为什么这样设计（对着三条核心要求）

- 简单可读：没有 RPC 框架、没有共享内存，协议就是文本，肉眼可调试——直接把请求文件 pipe 给后端进程就能单测。
- 性能影响可忽略：控制面与 token 流是 KB/s 量级，管道绰绰有余；GPU 计算完全不经过前端。
- 对 AI 开发友好：两侧各自独立可测——前端配一个 echo 假后端即可测协议，后端用脚本喂 stdin 即可测计算。

## 协议

交互契约（消息、时序、不变量）见 [protocol.md](protocol.md)。要点：NDJSON 文本，不使用 msgpack / pickle / 任何二进制序列化；stdout 只放协议数据；引擎进程常驻（模型加载是分钟级），并发由前端排队。

## 前端技术选型（2026-10）

- Python + FastAPI + uvicorn；日志一律 stderr。
- 对外提供 OpenAI（`/v1/chat/completions`）与 Anthropic（`/v1/messages`）两种 API 方言，流式 + 非流式；实现参考 `/home/teaonly/workspace/all-infer/Strata/serve` 的 server.py（方言端点、SSE 事件形态、/v1/v1 别名、Bearer/x-api-key）。
- 结构与 Strata 相反：它是 8700 行单文件 + stdlib http.server，myllm 按核心要求拆小模块。
- 本机 Python 3.14 无 pip / python3-venv：环境一律用 uv 建（`uv venv .venv && uv pip install -r requirements.txt`）。
- 图像落盘用内容寻址：sha256(原始字节) 作文件名（如 /var/tmp/myllm-img/<sha256>.<ext>）。同图同路径，天然去重、后端前缀缓存必命中；删除按 hash 引用计数归零执行（uid 开始 +1，finished/ErrorMsg −1），与协议的图像有效期规则对齐。
