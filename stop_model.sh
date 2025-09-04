#!/bin/bash

# ===========================
# vLLM 模型停止脚本
# ===========================

# 默认配置
DEFAULT_PORT=8000
DEFAULT_SERVED_MODEL_NAME="p2q"
LOG_DIR="logs"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 帮助信息
show_help() {
    echo "vLLM 模型停止脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -p, --port PORT            服务端口 (默认: $DEFAULT_PORT)"
    echo "  -n, --name NAME            服务模型名称 (默认: $DEFAULT_SERVED_MODEL_NAME)"
    echo "  --all                      停止所有 vLLM 服务"
    echo "  --force                    强制停止 (使用 SIGKILL)"
    echo "  --help                     显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0                                    # 停止默认端口的服务"
    echo "  $0 -p 8001                           # 停止指定端口的服务"
    echo "  $0 -n mymodel                        # 停止指定名称的服务"
    echo "  $0 --all                             # 停止所有 vLLM 服务"
    echo "  $0 --force                           # 强制停止服务"
    echo ""
}

# 解析命令行参数
PORT="$DEFAULT_PORT"
SERVED_MODEL_NAME="$DEFAULT_SERVED_MODEL_NAME"
STOP_ALL=false
FORCE_STOP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -n|--name)
            SERVED_MODEL_NAME="$2"
            shift 2
            ;;
        --all)
            STOP_ALL=true
            shift
            ;;
        --force)
            FORCE_STOP=true
            shift
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

# 停止指定端口的服务
stop_by_port() {
    local port=$1
    local force=$2
    
    echo -e "${YELLOW}正在查找端口 $port 上的 vLLM 服务...${NC}"
    
    # 查找占用端口的进程
    local pids=$(lsof -ti :$port 2>/dev/null)
    
    if [ -z "$pids" ]; then
        echo -e "${YELLOW}端口 $port 上没有运行的服务${NC}"
        return 0
    fi
    
    echo -e "找到进程: ${GREEN}$pids${NC}"
    
    # 显示进程信息
    for pid in $pids; do
        local cmd=$(ps -p $pid -o comm= 2>/dev/null)
        local args=$(ps -p $pid -o args= 2>/dev/null)
        echo -e "  PID $pid: $cmd"
        echo -e "  命令: $args"
    done
    
    # 确认停止
    if [ "$force" = false ]; then
        echo ""
        read -p "确认停止这些进程？(Y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Nn]$ ]]; then
            echo "停止操作已取消"
            return 0
        fi
    fi
    
    # 停止进程
    local stopped_count=0
    for pid in $pids; do
        if kill -0 $pid 2>/dev/null; then
            if [ "$force" = true ]; then
                echo -e "${YELLOW}强制停止进程 $pid...${NC}"
                kill -9 $pid 2>/dev/null
            else
                echo -e "${YELLOW}停止进程 $pid...${NC}"
                kill -TERM $pid 2>/dev/null
            fi
            
            # 等待进程结束
            local count=0
            while kill -0 $pid 2>/dev/null && [ $count -lt 10 ]; do
                sleep 1
                count=$((count + 1))
            done
            
            if kill -0 $pid 2>/dev/null; then
                if [ "$force" = false ]; then
                    echo -e "${YELLOW}进程 $pid 未响应，强制停止...${NC}"
                    kill -9 $pid 2>/dev/null
                fi
            fi
            
            if ! kill -0 $pid 2>/dev/null; then
                echo -e "${GREEN}✅ 进程 $pid 已停止${NC}"
                stopped_count=$((stopped_count + 1))
            else
                echo -e "${RED}❌ 进程 $pid 停止失败${NC}"
            fi
        fi
    done
    
    echo -e "${GREEN}已停止 $stopped_count 个进程${NC}"
    return 0
}

# 停止指定名称的服务
stop_by_name() {
    local name=$1
    local force=$2
    
    echo -e "${YELLOW}正在查找模型名称为 '$name' 的 vLLM 服务...${NC}"
    
    # 查找包含指定模型名称的 vLLM 进程
    local pids=$(pgrep -f "vllm.*--served-model-name.*$name" 2>/dev/null)
    
    if [ -z "$pids" ]; then
        echo -e "${YELLOW}没有找到模型名称为 '$name' 的服务${NC}"
        return 0
    fi
    
    echo -e "找到进程: ${GREEN}$pids${NC}"
    
    # 显示进程信息
    for pid in $pids; do
        local cmd=$(ps -p $pid -o comm= 2>/dev/null)
        local args=$(ps -p $pid -o args= 2>/dev/null)
        echo -e "  PID $pid: $cmd"
        echo -e "  命令: $args"
    done
    
    # 确认停止
    if [ "$force" = false ]; then
        echo ""
        read -p "确认停止这些进程？(Y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Nn]$ ]]; then
            echo "停止操作已取消"
            return 0
        fi
    fi
    
    # 停止进程
    local stopped_count=0
    for pid in $pids; do
        if kill -0 $pid 2>/dev/null; then
            if [ "$force" = true ]; then
                echo -e "${YELLOW}强制停止进程 $pid...${NC}"
                kill -9 $pid 2>/dev/null
            else
                echo -e "${YELLOW}停止进程 $pid...${NC}"
                kill -TERM $pid 2>/dev/null
            fi
            
            # 等待进程结束
            local count=0
            while kill -0 $pid 2>/dev/null && [ $count -lt 10 ]; do
                sleep 1
                count=$((count + 1))
            done
            
            if kill -0 $pid 2>/dev/null; then
                if [ "$force" = false ]; then
                    echo -e "${YELLOW}进程 $pid 未响应，强制停止...${NC}"
                    kill -9 $pid 2>/dev/null
                fi
            fi
            
            if ! kill -0 $pid 2>/dev/null; then
                echo -e "${GREEN}✅ 进程 $pid 已停止${NC}"
                stopped_count=$((stopped_count + 1))
            else
                echo -e "${RED}❌ 进程 $pid 停止失败${NC}"
            fi
        fi
    done
    
    echo -e "${GREEN}已停止 $stopped_count 个进程${NC}"
    return 0
}

# 停止所有 vLLM 服务
stop_all_vllm() {
    local force=$1
    
    echo -e "${YELLOW}正在查找所有 vLLM 服务...${NC}"
    
    # 查找所有 vLLM 进程
    local pids=$(pgrep -f "vllm" 2>/dev/null)
    
    if [ -z "$pids" ]; then
        echo -e "${YELLOW}没有找到运行中的 vLLM 服务${NC}"
        return 0
    fi
    
    echo -e "找到进程: ${GREEN}$pids${NC}"
    
    # 显示进程信息
    for pid in $pids; do
        local cmd=$(ps -p $pid -o comm= 2>/dev/null)
        local args=$(ps -p $pid -o args= 2>/dev/null)
        echo -e "  PID $pid: $cmd"
        echo -e "  命令: $args"
    done
    
    # 确认停止
    if [ "$force" = false ]; then
        echo ""
        read -p "确认停止所有 vLLM 服务？(Y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Nn]$ ]]; then
            echo "停止操作已取消"
            return 0
        fi
    fi
    
    # 停止进程
    local stopped_count=0
    for pid in $pids; do
        if kill -0 $pid 2>/dev/null; then
            if [ "$force" = true ]; then
                echo -e "${YELLOW}强制停止进程 $pid...${NC}"
                kill -9 $pid 2>/dev/null
            else
                echo -e "${YELLOW}停止进程 $pid...${NC}"
                kill -TERM $pid 2>/dev/null
            fi
            
            # 等待进程结束
            local count=0
            while kill -0 $pid 2>/dev/null && [ $count -lt 10 ]; do
                sleep 1
                count=$((count + 1))
            done
            
            if kill -0 $pid 2>/dev/null; then
                if [ "$force" = false ]; then
                    echo -e "${YELLOW}进程 $pid 未响应，强制停止...${NC}"
                    kill -9 $pid 2>/dev/null
                fi
            fi
            
            if ! kill -0 $pid 2>/dev/null; then
                echo -e "${GREEN}✅ 进程 $pid 已停止${NC}"
                stopped_count=$((stopped_count + 1))
            else
                echo -e "${RED}❌ 进程 $pid 停止失败${NC}"
            fi
        fi
    done
    
    echo -e "${GREEN}已停止 $stopped_count 个进程${NC}"
    return 0
}

# 清理 PID 文件
cleanup_pid_files() {
    echo -e "${YELLOW}清理 PID 文件...${NC}"
    
    if [ -d "$LOG_DIR" ]; then
        local pid_files=$(find "$LOG_DIR" -name "*.pid" 2>/dev/null)
        
        for pid_file in $pid_files; do
            if [ -f "$pid_file" ]; then
                local pid=$(cat "$pid_file" 2>/dev/null)
                if [ -n "$pid" ] && ! kill -0 $pid 2>/dev/null; then
                    echo -e "删除过期的 PID 文件: ${GREEN}$pid_file${NC}"
                    rm -f "$pid_file"
                fi
            fi
        done
    fi
}

# 主逻辑
main() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}        vLLM 模型停止脚本${NC}"
    echo -e "${BLUE}========================================${NC}"
    
    if [ "$STOP_ALL" = true ]; then
        echo -e "停止模式: ${GREEN}所有 vLLM 服务${NC}"
        stop_all_vllm $FORCE_STOP
    else
        echo -e "停止模式: ${GREEN}指定服务${NC}"
        echo -e "端口: ${GREEN}$PORT${NC}"
        echo -e "模型名称: ${GREEN}$SERVED_MODEL_NAME${NC}"
        
        # 先按端口停止
        stop_by_port $PORT $FORCE_STOP
        
        # 再按名称停止（可能会有遗漏）
        stop_by_name $SERVED_MODEL_NAME $FORCE_STOP
    fi
    
    # 清理 PID 文件
    cleanup_pid_files
    
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}           停止完成${NC}"
    echo -e "${BLUE}========================================${NC}"
    
    # 显示剩余进程
    local remaining=$(pgrep -f "vllm" 2>/dev/null)
    if [ -n "$remaining" ]; then
        echo -e "${YELLOW}⚠️  仍有 vLLM 进程在运行:${NC}"
        for pid in $remaining; do
            local cmd=$(ps -p $pid -o comm= 2>/dev/null)
            local args=$(ps -p $pid -o args= 2>/dev/null)
            echo -e "  PID $pid: $cmd"
        done
        echo -e "使用 ${GREEN}--force${NC} 选项强制停止"
    else
        echo -e "${GREEN}✅ 所有 vLLM 服务已停止${NC}"
    fi
}

# 运行主函数
main
