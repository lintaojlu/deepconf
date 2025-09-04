#!/bin/bash

# vLLM 强制安装脚本 (跳过 Git 检测)
echo "🔧 强制安装 vLLM (跳过 Git 检测)..."

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

# 安装必要的构建依赖
echo "📦 安装构建依赖..."
uv pip install setuptools-scm wheel build

cd vllm

# 设置环境变量完全跳过 Git 检测
export SETUPTOOLS_SCM_PRETEND_VERSION=1.0.0
export SETUPTOOLS_SCM_IGNORE_VCS_ROOTS="*"
export SETUPTOOLS_SCM_DISABLE=true

echo "📦 正在安装 vLLM..."
echo "⚠️  跳过所有 Git 和版本检测..."

# 检测系统类型，决定安装方式
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🍎 检测到 macOS，使用源码编译..."
    SETUPTOOLS_SCM_PRETEND_VERSION=1.0.0 uv pip install --editable . --force-reinstall
else
    echo "🐧 检测到 Linux，使用预编译版本..."
    export VLLM_USE_PRECOMPILED=1
    VLLM_USE_PRECOMPILED=1 SETUPTOOLS_SCM_PRETEND_VERSION=1.0.0 uv pip install --editable . --force-reinstall
fi

if [ $? -eq 0 ]; then
    echo "✅ vLLM 安装成功！"
    cd ..
    
    # 验证安装
    echo "🧪 验证 vLLM 安装..."
    python -c "import vllm; print(f'✅ vLLM 版本: {vllm.__version__}')" || echo "❌ vLLM 导入失败"
else
    echo "❌ vLLM 安装失败"
    cd ..
    exit 1
fi
