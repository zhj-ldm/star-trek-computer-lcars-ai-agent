#!/bin/bash
# ===== Star Trek 语音助手一键启动（语音识别 + Web 界面） =====
# 用法: ./start_all.sh
# 关闭本终端窗口或按 Ctrl+C 即停止全部服务
# 模型/密钥/API地址/唤醒词/阈值等配置统一在 config.json 中修改
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

BASE="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE"

PY=~/agentscope-env/bin/python

# 启动语音助手
"$PY" voice_assistant.py &
VOICE_PID=$!

# 启动 Web 界面
"$PY" web_ui.py &
WEB_PID=$!

# 关闭窗口 / Ctrl+C 时同时结束两个进程
cleanup() {
  kill "$VOICE_PID" "$WEB_PID" 2>/dev/null
  wait 2>/dev/null
  exit 0
}
trap cleanup INT TERM EXIT

# 任一进程退出则整体退出
while kill -0 "$VOICE_PID" 2>/dev/null && kill -0 "$WEB_PID" 2>/dev/null; do
  sleep 1
done
cleanup
