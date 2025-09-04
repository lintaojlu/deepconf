#!/bin/bash

# DeepConf 项目环境设置脚本 (不包含 vllm)
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
    if [ -f "$HOME/.cargo/env" ]; then
        source "$HOME/.cargo/env"
    fi
else
    echo "✅ uv 已安装: $(uv --version)"
fi

# 创建虚拟环境并安装依赖
echo "🔧 创建虚拟环境并安装依赖..."
uv venv .venv
source .venv/bin/activate
uv pip install -e .

# 验证环境
echo "🧪 验证环境..."
python -c "import torch; print(f'✅ PyTorch: {torch.__version__}')" || echo "❌ PyTorch 导入失败"
python -c "import pandas; print('✅ Pandas: OK')" || echo "❌ Pandas 导入失败"
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
echo "💡 如果需要 vllm，可以稍后手动安装："
echo "   cd vllm && VLLM_USE_PRECOMPILED=1 uv pip install --editable ."
echo ""
echo "🚀 开始使用 DeepConf 项目吧！"
