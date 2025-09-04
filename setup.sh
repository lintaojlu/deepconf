#!/bin/bash

# DeepConf 项目环境设置脚本
echo "🚀 开始设置 DeepConf 项目环境..."

# 检查 Python 版本
echo "📋 检查 Python 版本..."
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✅ Python 版本: $(python3 --version) (满足要求 >=3.10)"
else
    echo "❌ Python 版本过低: $(python3 --version)"
    echo "请安装 Python 3.10 或更高版本"
    exit 1
fi

# 检查 uv 是否已安装
if ! command -v uv &> /dev/null; then
    echo "📦 安装 uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
else
    echo "✅ uv 已安装: $(uv --version)"
fi

# 创建虚拟环境并安装依赖
echo "🔧 创建虚拟环境并安装依赖..."
uv venv .venv
source .venv/bin/activate
uv pip install -e .

# 安装 vllm
echo "🔧 安装 vllm..."
if [ -d "vllm" ]; then
    # 先安装构建依赖
    echo "📦 安装构建依赖..."
    uv pip install setuptools-scm wheel build
    
    cd vllm
    echo "📦 正在安装 vllm (跳过 Git 检测)..."
    
    # 设置环境变量跳过 Git 检测
    export SETUPTOOLS_SCM_PRETEND_VERSION=1.0.0
    export SETUPTOOLS_SCM_IGNORE_VCS_ROOTS="*"
    
    # 检测系统类型，决定是否使用预编译版本
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "🍎 检测到 macOS，使用源码编译..."
        SETUPTOOLS_SCM_PRETEND_VERSION=1.0.0 uv pip install --editable .
    else
        echo "🐧 检测到 Linux，使用预编译版本..."
        export VLLM_USE_PRECOMPILED=1
        VLLM_USE_PRECOMPILED=1 SETUPTOOLS_SCM_PRETEND_VERSION=1.0.0 uv pip install --editable .
    fi
    
    echo "✅ vllm 安装完成"
    cd ..
else
    echo "❌ vllm 目录不存在"
    exit 1
fi

# 验证环境
echo "🧪 验证环境..."
python -c "import torch; print(f'✅ PyTorch: {torch.__version__}')" || echo "❌ PyTorch 导入失败"
python -c "import pandas; print('✅ Pandas: OK')" || echo "❌ Pandas 导入失败"
python -c "import vllm; print('✅ vLLM: OK')" || echo "❌ vLLM 导入失败"
python -c "import sentence_transformers; print('✅ Sentence Transformers: OK')" || echo "❌ Sentence Transformers 导入失败"

echo ""
echo "🎉 环境设置完成！"
echo ""
echo "📝 使用说明："
echo "1. 激活环境: source .venv/bin/activate"
echo "2. 运行脚本: python infer/deepconf_online_generation.py"
echo "3. 添加依赖: uv add package_name"
echo "4. 同步依赖: uv sync"
echo ""
echo "🚀 开始使用 DeepConf 项目吧！"
