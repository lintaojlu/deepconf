#!/bin/bash

# ===========================
# vLLM 模型部署脚本
# ===========================

# 默认配置
DEFAULT_MODEL_PATH="checkpoint-192"
DEFAULT_SERVED_MODEL_NAME="p2q"
DEFAULT_PORT=8000
DEFAULT_DTYPE="half"
DEFAULT_GPU_MEMORY_UTIL=0.7
DEFAULT_TENSOR_PARALLEL_SIZE=1
DEFAULT_HOST="0.0.0.0"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 帮助信息
show_help() {
    echo "vLLM 模型部署脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -m, --model PATH           模型路径 (默认: $DEFAULT_MODEL_PATH)"
    echo "  -n, --name NAME            服务模型名称 (默认: $DEFAULT_SERVED_MODEL_NAME)"
    echo "  -p, --port PORT            服务端口 (默认: $DEFAULT_PORT)"
    echo "  -d, --dtype DTYPE          数据类型 (默认: $DEFAULT_DTYPE)"
    echo "  -g, --gpu-memory UTIL      GPU内存使用率 (默认: $DEFAULT_GPU_MEMORY_UTIL)"
    echo "  -t, --tensor-parallel SIZE 张量并行大小 (默认: $DEFAULT_TENSOR_PARALLEL_SIZE)"
    echo "  -h, --host HOST            监听地址 (默认: $DEFAULT_HOST)"
    echo "  --help                     显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0                                    # 使用默认配置"
    echo "  $0 -m /path/to/model -n mymodel      # 自定义模型路径和名称"
    echo "  $0 -p 8001 -d float16                # 自定义端口和数据类型"
    echo ""
}

# 解析命令行参数
MODEL_PATH="$DEFAULT_MODEL_PATH"
SERVED_MODEL_NAME="$DEFAULT_SERVED_MODEL_NAME"
PORT="$DEFAULT_PORT"
DTYPE="$DEFAULT_DTYPE"
GPU_MEMORY_UTIL="$DEFAULT_GPU_MEMORY_UTIL"
TENSOR_PARALLEL_SIZE="$DEFAULT_TENSOR_PARALLEL_SIZE"
HOST="$DEFAULT_HOST"

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL_PATH="$2"
            shift 2
            ;;
        -n|--name)
            SERVED_MODEL_NAME="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -d|--dtype)
            DTYPE="$2"
            shift 2
            ;;
        -g|--gpu-memory)
            GPU_MEMORY_UTIL="$2"
            shift 2
            ;;
        -t|--tensor-parallel)
            TENSOR_PARALLEL_SIZE="$2"
            shift 2
            ;;
        -h|--host)
            HOST="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}错误: 未知参数 $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# 检查模型路径是否存在
if [ ! -d "$MODEL_PATH" ] && [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}错误: 模型路径 '$MODEL_PATH' 不存在${NC}"
    exit 1
fi

# 检查端口是否被占用
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${YELLOW}警告: 端口 $PORT 已被占用${NC}"
    echo "正在运行的进程:"
    lsof -Pi :$PORT -sTCP:LISTEN
    echo ""
    read -p "是否继续？这可能会覆盖现有服务 (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "部署已取消"
        exit 1
    fi
fi

# 检查 vLLM 是否安装
if ! python -c "import vllm" 2>/dev/null; then
    echo -e "${RED}错误: vLLM 未安装或未正确配置${NC}"
    echo "请先安装 vLLM:"
    echo "  pip install vllm"
    echo "或者运行项目设置脚本:"
    echo "  ./setup.sh"
    exit 1
fi

# 创建日志目录
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# 生成日志文件名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/vllm_${SERVED_MODEL_NAME}_${PORT}_${TIMESTAMP}.log"
PID_FILE="$LOG_DIR/vllm_${SERVED_MODEL_NAME}_${PORT}.pid"

# 显示配置信息
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}        vLLM 模型部署配置${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "模型路径: ${GREEN}$MODEL_PATH${NC}"
echo -e "服务名称: ${GREEN}$SERVED_MODEL_NAME${NC}"
echo -e "监听地址: ${GREEN}$HOST:$PORT${NC}"
echo -e "数据类型: ${GREEN}$DTYPE${NC}"
echo -e "GPU内存使用率: ${GREEN}$GPU_MEMORY_UTIL${NC}"
echo -e "张量并行大小: ${GREEN}$TENSOR_PARALLEL_SIZE${NC}"
echo -e "日志文件: ${GREEN}$LOG_FILE${NC}"
echo -e "PID文件: ${GREEN}$PID_FILE${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 构建 vLLM 启动命令
VLLM_CMD="python -m vllm.entrypoints.openai.api_server"
VLLM_CMD="$VLLM_CMD --model $MODEL_PATH"
VLLM_CMD="$VLLM_CMD --served-model-name $SERVED_MODEL_NAME"
VLLM_CMD="$VLLM_CMD --host $HOST"
VLLM_CMD="$VLLM_CMD --port $PORT"
VLLM_CMD="$VLLM_CMD --dtype $DTYPE"
VLLM_CMD="$VLLM_CMD --gpu-memory-utilization $GPU_MEMORY_UTIL"
VLLM_CMD="$VLLM_CMD --tensor-parallel-size $TENSOR_PARALLEL_SIZE"
VLLM_CMD="$VLLM_CMD --enable-prefix-caching"

echo -e "${YELLOW}启动命令:${NC}"
echo "$VLLM_CMD"
echo ""

# 确认启动
read -p "确认启动模型服务？(Y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Nn]$ ]]; then
    echo "部署已取消"
    exit 0
fi

# 启动服务
echo -e "${GREEN}正在启动 vLLM 服务...${NC}"
echo "启动时间: $(date)"
echo ""

# 后台启动服务
nohup $VLLM_CMD > "$LOG_FILE" 2>&1 &
VLLM_PID=$!

# 保存 PID
echo $VLLM_PID > "$PID_FILE"

echo -e "${GREEN}✅ vLLM 服务已启动${NC}"
echo -e "进程ID: ${GREEN}$VLLM_PID${NC}"
echo -e "日志文件: ${GREEN}$LOG_FILE${NC}"
echo -e "PID文件: ${GREEN}$PID_FILE${NC}"
echo ""

# 等待服务启动
echo -e "${YELLOW}等待服务启动...${NC}"
sleep 5

# 检查服务是否正常运行
if kill -0 $VLLM_PID 2>/dev/null; then
    echo -e "${GREEN}✅ 服务进程运行正常${NC}"
    
    # 测试 API 连接
    echo -e "${YELLOW}测试 API 连接...${NC}"
    if curl -s "http://$HOST:$PORT/v1/models" >/dev/null 2>&1; then
        echo -e "${GREEN}✅ API 服务正常响应${NC}"
        echo -e "API 地址: ${GREEN}http://$HOST:$PORT/v1${NC}"
    else
        echo -e "${YELLOW}⚠️  API 服务可能还在启动中，请稍后检查${NC}"
    fi
else
    echo -e "${RED}❌ 服务启动失败${NC}"
    echo "请检查日志文件: $LOG_FILE"
    exit 1
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}           部署完成${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "服务名称: ${GREEN}$SERVED_MODEL_NAME${NC}"
echo -e "API 地址: ${GREEN}http://$HOST:$PORT/v1${NC}"
echo -e "进程ID: ${GREEN}$VLLM_PID${NC}"
echo -e "日志文件: ${GREEN}$LOG_FILE${NC}"
echo ""
echo -e "${YELLOW}管理命令:${NC}"
echo -e "查看日志: ${GREEN}tail -f $LOG_FILE${NC}"
echo -e "停止服务: ${GREEN}./stop_model.sh -p $PORT${NC}"
echo -e "检查状态: ${GREEN}ps aux | grep $VLLM_PID${NC}"
echo ""
echo -e "${GREEN}🎉 模型部署成功！${NC}"
