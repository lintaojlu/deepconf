#!/bin/bash

# 测试 vLLM 安装脚本
echo "🧪 测试 vLLM 安装..."

# 检查是否在正确的目录
if [ ! -d "vllm" ]; then
    echo "❌ 错误: 请在项目根目录运行此脚本"
    exit 1
fi

# 激活虚拟环境
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✅ 已激活虚拟环境"
else
    echo "❌ 错误: 虚拟环境不存在，请先运行 setup.sh"
    exit 1
fi

# 安装构建依赖
echo "📦 安装构建依赖..."
uv pip install setuptools-scm wheel build

cd vllm

# 设置环境变量
export SETUPTOOLS_SCM_PRETEND_VERSION=1.0.0
export SETUPTOOLS_SCM_IGNORE_VCS_ROOTS="*"

echo "📦 测试安装 vLLM (dry-run)..."

# 测试安装（dry-run）
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🍎 检测到 macOS，测试源码编译..."
    SETUPTOOLS_SCM_PRETEND_VERSION=1.0.0 uv pip install --editable . --dry-run
else
    echo "🐧 检测到 Linux，测试预编译版本..."
    export VLLM_USE_PRECOMPILED=1
    VLLM_USE_PRECOMPILED=1 SETUPTOOLS_SCM_PRETEND_VERSION=1.0.0 uv pip install --editable . --dry-run
fi

if [ $? -eq 0 ]; then
    echo "✅ vLLM 安装测试通过！"
    echo "💡 现在可以运行 ./install-vllm.sh 进行实际安装"
else
    echo "❌ vLLM 安装测试失败"
fi

cd ..
