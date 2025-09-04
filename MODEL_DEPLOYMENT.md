# vLLM 模型部署脚本使用说明

## 📋 概述

本项目提供了两个脚本来管理 vLLM 模型的部署和停止：

- `deploy_model.sh` - 模型部署脚本
- `stop_model.sh` - 模型停止脚本

## 🚀 模型部署脚本 (`deploy_model.sh`)

### 基本用法

```bash
# 使用默认配置部署
./deploy_model.sh

# 自定义模型路径和名称
./deploy_model.sh -m /path/to/your/model -n your_model_name

# 自定义端口和数据类型
./deploy_model.sh -p 8001 -d float16
```

### 参数说明

| 参数 | 长参数 | 说明 | 默认值 |
|------|--------|------|--------|
| `-m` | `--model` | 模型路径 | `checkpoint-192` |
| `-n` | `--name` | 服务模型名称 | `p2q` |
| `-p` | `--port` | 服务端口 | `8000` |
| `-d` | `--dtype` | 数据类型 | `half` |
| `-g` | `--gpu-memory` | GPU内存使用率 | `0.7` |
| `-t` | `--tensor-parallel` | 张量并行大小 | `1` |
| `-h` | `--host` | 监听地址 | `0.0.0.0` |
| | `--help` | 显示帮助信息 | |

### 使用示例

```bash
# 1. 基本部署
./deploy_model.sh

# 2. 部署自定义模型
./deploy_model.sh -m /models/llama2-7b -n llama2

# 3. 多GPU部署
./deploy_model.sh -t 2 -g 0.8

# 4. 自定义端口
./deploy_model.sh -p 8001 -n mymodel

# 5. 查看帮助
./deploy_model.sh --help
```

### 功能特性

- ✅ **自动检查**: 检查模型路径、端口占用、vLLM安装
- ✅ **后台运行**: 使用 nohup 后台运行服务
- ✅ **日志记录**: 自动生成日志文件
- ✅ **PID管理**: 保存进程ID便于管理
- ✅ **健康检查**: 自动测试API连接
- ✅ **彩色输出**: 友好的命令行界面

## 🛑 模型停止脚本 (`stop_model.sh`)

### 基本用法

```bash
# 停止默认端口的服务
./stop_model.sh

# 停止指定端口的服务
./stop_model.sh -p 8001

# 停止指定名称的服务
./stop_model.sh -n mymodel

# 停止所有vLLM服务
./stop_model.sh --all

# 强制停止服务
./stop_model.sh --force
```

### 参数说明

| 参数 | 长参数 | 说明 | 默认值 |
|------|--------|------|--------|
| `-p` | `--port` | 服务端口 | `8000` |
| `-n` | `--name` | 服务模型名称 | `p2q` |
| | `--all` | 停止所有vLLM服务 | |
| | `--force` | 强制停止(SIGKILL) | |
| | `--help` | 显示帮助信息 | |

### 使用示例

```bash
# 1. 停止默认服务
./stop_model.sh

# 2. 停止指定端口服务
./stop_model.sh -p 8001

# 3. 停止指定模型服务
./stop_model.sh -n llama2

# 4. 停止所有vLLM服务
./stop_model.sh --all

# 5. 强制停止服务
./stop_model.sh --force

# 6. 查看帮助
./stop_model.sh --help
```

### 功能特性

- ✅ **多种停止方式**: 按端口、按名称、全部停止
- ✅ **安全停止**: 先发送TERM信号，再发送KILL信号
- ✅ **进程检查**: 显示进程信息，确认停止
- ✅ **PID清理**: 自动清理过期的PID文件
- ✅ **状态报告**: 显示停止结果和剩余进程

## 📁 文件结构

部署后会在项目根目录创建以下文件：

```
deepconf/
├── deploy_model.sh          # 部署脚本
├── stop_model.sh            # 停止脚本
├── logs/                    # 日志目录
│   ├── vllm_p2q_8000_20241204_164500.log  # 服务日志
│   └── vllm_p2q_8000.pid                  # 进程ID文件
└── MODEL_DEPLOYMENT.md      # 本文档
```

## 🔧 管理命令

### 查看服务状态

```bash
# 查看进程
ps aux | grep vllm

# 查看端口占用
lsof -i :8000

# 查看日志
tail -f logs/vllm_p2q_8000_*.log
```

### 测试API连接

```bash
# 测试模型列表
curl http://localhost:8000/v1/models

# 测试聊天完成
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "p2q",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

## ⚠️ 注意事项

1. **端口冲突**: 确保端口未被其他服务占用
2. **GPU内存**: 根据GPU显存调整内存使用率
3. **模型路径**: 确保模型路径正确且可访问
4. **权限问题**: 确保脚本有执行权限
5. **依赖检查**: 确保vLLM已正确安装

## 🐛 故障排除

### 常见问题

1. **端口被占用**
   ```bash
   # 查看占用进程
   lsof -i :8000
   # 停止占用进程
   ./stop_model.sh -p 8000
   ```

2. **模型路径错误**
   ```bash
   # 检查模型路径
   ls -la /path/to/your/model
   ```

3. **vLLM未安装**
   ```bash
   # 安装vLLM
   pip install vllm
   # 或运行项目设置
   ./setup.sh
   ```

4. **GPU内存不足**
   ```bash
   # 降低GPU内存使用率
   ./deploy_model.sh -g 0.5
   ```

### 日志分析

```bash
# 查看最新日志
tail -f logs/vllm_*.log

# 搜索错误信息
grep -i error logs/vllm_*.log

# 查看启动信息
grep -i "starting\|started" logs/vllm_*.log
```

## 📞 支持

如果遇到问题，请：

1. 查看日志文件
2. 检查系统资源
3. 确认配置参数
4. 参考vLLM官方文档

---

**Happy Deploying! 🚀**
