# 架构：前后端分离，stdio 交互（CGI 风格）

```
客户端 ──HTTP──> HTTP API 进程（前端） ──stdin/stdout, fd 0/1──> GPU 推理进程（后端）
```

## 职责

- **前端（HTTP API）**：面向客户端的 HTTP 服务，负责请求解析、tokenize / detokenize、会话与并发管理、流式输出（SSE / chunked）；不碰 GPU。
- **后端（GPU 推理进程）**：加载模型，专注 prefill/decode 计算；不做网络 IO。
- **交互**：前端把请求写进后端的 stdin，后端逐 token 写回 stdout。两个独立进程，边界即模块边界。

## 为什么这样设计（对着三条核心要求）

- 简单可读：没有 RPC 框架、没有共享内存，协议就是文本，肉眼可调试——直接把请求文件 pipe 给后端进程就能单测。
- 性能影响可忽略：控制面与 token 流是 KB/s 量级，管道绰绰有余；GPU 计算完全不经过前端。
- 对 AI 开发友好：两侧各自独立可测——前端配一个 echo 假后端即可测协议，后端用脚本喂 stdin 即可测计算。

## 协议

交互契约（消息、时序、不变量）见 [protocol.md](protocol.md)。要点：NDJSON 文本，不使用 msgpack / pickle / 任何二进制序列化；stdout 只放协议数据；引擎进程常驻（模型加载是分钟级），并发由前端排队。
