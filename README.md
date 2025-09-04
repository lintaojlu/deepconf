# DeepConf - Deep Confidence Learning Project

深度置信学习项目，包含在线生成和投票算法实现。

## 🚀 快速开始

### 新机器环境搭建

#### 方法1：一键安装（推荐）

```bash
git clone <your-repo-url>
cd deepconf
./setup.sh
```

#### 方法2：手动设置

1. **检查 Python 版本** (要求 >= 3.10)
```bash
python --version
```

2. **安装 uv**
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
irm https://astral.sh/uv/install.ps1 | iex
```

3. **克隆项目并安装依赖**
```bash
git clone <your-repo-url>
cd deepconf
uv venv .venv
source .venv/bin/activate  # macOS/Linux
# 或 .venv\Scripts\activate  # Windows
uv pip install -e .

# 安装 vLLM (使用克隆方式)
./install-vllm-clone.sh
```

## 📁 项目结构

```
deepconf/
├── infer/                           # 主要代码目录
│   ├── deepconf_online_generation.py    # 在线生成算法
│   ├── deepconf_offline_generation.py   # 离线生成算法
│   ├── deepconf_offline_voting.py       # 离线投票算法
│   ├── sentence_transformer_utils.py    # 语义相似度工具
│   └── data/                            # 数据文件 (Git 忽略)
├── vllm/                           # vLLM 仓库 (Git 忽略，自动克隆)
├── pyproject.toml                  # 项目配置和依赖
├── requirements.txt                # 传统依赖文件
├── setup.sh                       # 完整环境设置脚本
├── install-vllm-clone.sh          # vLLM 克隆安装脚本
└── .venv/                         # 虚拟环境
```

## 🎯 使用方法

### 运行主要脚本

```bash
# 激活环境
source .venv/bin/activate  # macOS/Linux
# 或 .venv\Scripts\activate  # Windows

# 运行在线生成
python infer/deepconf_online_generation.py

# 运行离线生成
python infer/deepconf_offline_generation.py

# 运行离线投票
python infer/deepconf_offline_voting.py
```

### 依赖管理

```bash
# 添加新依赖
uv add package_name

# 安装开发依赖
uv add pytest black --dev

# 同步依赖
uv sync

# 查看依赖树
uv tree
```

## 🔧 环境要求

- **Python**: >= 3.10
- **主要依赖**:
  - PyTorch >= 2.7.1
  - sentence-transformers >= 2.2.0
  - transformers >= 4.21.0
  - pandas >= 1.5.0
  - numpy >= 1.21.0
  - vLLM (从 GitHub 克隆安装)

## 📝 开发说明

- 项目使用 `uv` 进行依赖管理
- `pyproject.toml` 定义依赖和配置
- `requirements.txt` 提供传统 pip 兼容性
- vLLM 使用克隆方式安装，包含特定 PR 分支
- 使用灵活的版本管理（无严格锁定）

## 🆘 常见问题

### Q: 如何在新机器上快速搭建环境？
A: 使用 `./setup.sh` 一键设置。

### Q: vLLM 安装失败怎么办？
A: 可以单独运行 `./install-vllm-clone.sh` 重新安装 vLLM。

### Q: 如何添加新的 Python 依赖？
A: 使用 `uv add package_name` 命令。

### Q: 支持哪些 Python 版本？
A: 要求 Python >= 3.10，推荐使用 Python 3.11 或 3.12。

### Q: vLLM 使用哪个版本？
A: 使用 GitHub 上的特定 PR 分支 (pull/23201/head)。

## 🔄 故障排除

### vLLM 安装问题
如果 vLLM 安装失败，可以：
1. 删除 vllm 目录：`rm -rf vllm`
2. 重新安装：`./install-vllm-clone.sh`

### Shell 兼容性问题
如果遇到 `source: not found` 错误，请使用：
```bash
. .venv/bin/activate  # 而不是 source .venv/bin/activate
```

### Git 相关问题
确保系统已安装 Git：
```bash
git --version
```