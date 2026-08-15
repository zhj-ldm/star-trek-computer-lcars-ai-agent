---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: abc280594deb8e830e487cdb027f0632_287ece0898b111f18cca525400e6dd8f
    ReservedCode1: ZqvaQuCueiJhqLCkTJDHuCwvgLfMkM0HM2eE2tzSnh1Z0D/fGfg4FqNQ7Mnl5LJFAX1sgQPkNZAiCujCxPfXkiccfWExBSl3L+D2e25HQl7VLPi1SQB3uMJfIoSPJpl7Bj1Fm51TsfdoJT8MST2z062h0zTppgQFMN8d2cxTUgtmdWFkyn+XzaDV1Xk=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: abc280594deb8e830e487cdb027f0632_287ece0898b111f18cca525400e6dd8f
    ReservedCode2: ZqvaQuCueiJhqLCkTJDHuCwvgLfMkM0HM2eE2tzSnh1Z0D/fGfg4FqNQ7Mnl5LJFAX1sgQPkNZAiCujCxPfXkiccfWExBSl3L+D2e25HQl7VLPi1SQB3uMJfIoSPJpl7Bj1Fm51TsfdoJT8MST2z062h0zTppgQFMN8d2cxTUgtmdWFkyn+XzaDV1Xk=
---

# Star Trek Computer 项目架构介绍

> 星际迷航风格本地语音助手：唤醒即对话，支持联网搜索、文件操作、终端命令与 Web 管理界面。

---

## 1. 项目概述与定位

**Star Trek Computer** 是一个运行在 macOS 上的**本地语音助手**，以《星际迷航》中舰载电脑的交互方式为设计蓝本：用户说出唤醒词（如 `computer`）即可唤起助手，随后用自然语言下达指令，助手通过语音回复结果。

项目核心定位：

- **纯本地语音链路**：唤醒词检测、录音、语音识别（ASR）、语音合成（TTS）均在本地完成，仅 LLM 推理调用云端 OpenAI 兼容 API。
- **Agent 能力**：基于 AgentScope 框架构建 Agent，具备联网搜索、网页抓取、文件读写、终端命令执行等工具能力。
- **多形态入口**：语音对话（主链路）、Flask Web 管理界面、tkinter 桌面聊天窗口三种交互方式。
- **多会话 + 分层记忆**：支持多对话管理，历史消息自动摘要压缩与关键词检索，对话持久化到本地 JSON 文件。

---

## 2. 整体架构

### 2.1 模块划分

```
Star Trek computer/
├── voice_assistant.py   # 主程序：语音链路核心（唤醒/录音/转录/LLM/TTS/权限）
├── voice_chat.py        # 桌面聊天 GUI（tkinter）+ Agent 工具集构建
├── web_ui.py            # Flask Web 管理界面（复用 LLMHandler）
├── start_all.sh         # 一键启动（语音 + Web）
├── config.json          # 全局配置（模型/唤醒词/录音/记忆）
├── requirements.txt     # 依赖清单
├── templates/
│   ├── chat.html        # Web 聊天页（星际迷航风格 canvas 渲染）
│   └── setup.html       # Web 设置页
├── 对话/                # 多会话持久化目录（对话N.json）
├── hey_computer.onnx    # 唤醒词模型（旧版）
├── computer_v2.onnx     # 唤醒词模型（当前 config 使用）
├── wake_sound.wav       # 唤醒提示音
├── complete.mp3         # 发送给 AI 时的音效
├── alert23.mp3          # AI 出错音效
└── commandfunctionsoffline_ep.mp3  # AI 掉线音效
```

### 2.2 语音链路数据流

```
麦克风持续监听
   │  (sounddevice InputStream, 16kHz)
   ▼
openwakeword 唤醒词检测（onnx 推理）
   │  score >= wakeword_threshold（默认 0.15）
   ▼
播放 wake_sound.wav 提示音
   │
   ▼
录音（RMS 能量 VAD：静音 1.5s 结束，最长 25s）
   │
   ▼
faster-whisper small 转录（beam_size=5 + vad_filter + 语言自动检测）
   │
   ▼
播放 complete.mp3（开始发送给 AI）
   │
   ▼
AgentScope Agent 执行（LLM + 工具调用）
   │  web_search / fetch_page / Read / Write / Glob / Grep / Bash
   ▼
回复文本按句切分 → Edge TTS 合成 mp3 → afplay 播放
   │
   ├─ 成功：正常朗读回复
   └─ 失败/掉线/无回复：播放 alert23.mp3 + commandfunctionsoffline_ep.mp3
```

### 2.3 Web 界面数据流

```
浏览器（chat.html / setup.html）
   │  fetch /api/*  +  EventSource /api/stream（SSE）
   ▼
Flask（web_ui.py）
   │  复用 voice_assistant.LLMHandler（多会话 + 分层记忆 + 持久化）
   ▼
AgentScope Agent 执行
   │  工具日志 / 回复 / 删除确认事件 → SSE 广播
   ▼
浏览器实时渲染（星际迷航风格 UI）
```

---

## 3. 各模块/文件功能说明

### 3.1 voice_assistant.py（主程序，约 1100 行）

语音链路核心，包含以下主要组件：

| 组件 | 说明 |
|---|---|
| `EdgeTTSSynthesizer` | Edge TTS 合成 mp3 后用 `afplay` 播放的后台队列线程；支持打断（interrupt_event）、清空队列、自动中/英文音色选择 |
| `LLMHandler` | 多会话管理 + 分层记忆 + 对话持久化；负责 Agent 构建、历史摘要、关键词检索、对话增删切换 |
| `VoiceAssistant` | 主循环：唤醒词监听 → 提示音 → 录音 → 转录 → 处理 → 重开流；含 Cmd+Shift+M 打断监听 |
| `VoicePermissionMiddleware` | 分级权限中间件：只读工具/非删除命令直接放行，仅删除类 Bash 命令转语音确认 |
| `build_agent` | 组装 AgentScope Agent（OpenAICredential + OpenAIChatModel + Toolkit + 中间件） |
| `parse_args` / `main` | 命令行入口，支持 `--list-devices`、`--list-voices`、`--debug` 等参数 |

**关键常量**：`SAMPLE_RATE=16000`、`SILENCE_THRESHOLD=400`、`SILENCE_SEC=1.5`、`MAX_RECORD_SEC=25`、`WAKE_THRESHOLD=0.15`。

**分层记忆机制**（LLMHandler）：
- **近期消息**：保留最近 `recent_keep=6` 条完整对话。
- **摘要压缩**：历史超过 `summary_trigger=12` 条时，将久远部分用模型压缩进摘要，仅保留最近消息。
- **关键词检索**：按关键词重叠召回与当前问题最相关的历史片段（`retrieve_top=3`），注入 system prompt。

**对话管理**：支持语音命令「创建对话 / 切换对话 / 列出对话 / 清空对话」，多会话持久化到 `对话/` 目录（`对话N.json`），以磁盘文件为唯一真相（`_resync_from_disk`）。

**音效反馈**：
- 发送给 AI 时播放 `complete.mp3`；
- AI 出错/掉线/无回复时依次播放 `alert23.mp3`、`commandfunctionsoffline_ep.mp3`。

### 3.2 voice_chat.py（桌面聊天 GUI + 工具集）

- **`build_toolkit()`**：组装 Agent 工具集——`web_search`（多引擎联网搜索）、`fetch_page`（网页抓取）、`Read` / `Write` / `Glob` / `Grep`（文件操作）、`Bash`（终端命令）。
- **`VoiceChatApp`**：tkinter 桌面聊天窗口，支持任意 OpenAI 兼容端点（模型名/Base URL/API Key 可填），实时显示工具执行日志（灰色小字）。
- **`SYSTEM_PROMPT`**：定义助手系统提示词，声明其联网搜索、网页抓取、文件操作、终端命令能力。

### 3.3 web_ui.py（Flask Web 管理界面）

- 复用 `voice_assistant.py` 的 `LLMHandler` / `build_agent` / `load_config`，共享 `config.json` 与 `对话/` 目录。
- **页面路由**：`/` 与 `/setup`（设置页）、`/chat`（聊天页）。
- **配置接口**：`/api/config`（GET）、`/api/config/model`、`/api/config/audio`、`/api/config/memory`（POST）。
- **对话接口**：`/api/chats`、`/api/chats/new`、`/api/chats/<name>/history`、`/api/chats/<name>/send`。
- **删除确认接口**：`/api/confirm/<ok>`（前端弹窗确认/取消）。
- **SSE 事件流**：`/api/stream`，广播用户消息、AI 回复、工具日志、确认弹窗等事件。

### 3.4 启动脚本

| 脚本 | 功能 |
|---|---|
| `start_all.sh` | 一键启动：同时启动语音助手（voice_assistant.py）与 Web 界面（web_ui.py），任一退出则整体退出 |

脚本特性：`BASE` 动态获取脚本所在目录（`$(cd "$(dirname "$0")" && pwd)`），项目改名/移动不影响运行；使用 `~/agentscope-env/bin/python` 虚拟环境解释器；设置 `HF_ENDPOINT=https://hf-mirror.com` 与 `HF_HUB_DISABLE_XET=1` 加速模型下载。

### 3.5 config.json（全局配置）

| 配置项 | 说明 | 当前值 |
|---|---|---|
| `model` | OpenAI 兼容模型名 | 空（需配置） |
| `api_key` | API Key | 空（需配置） |
| `base_url` | OpenAI 兼容 Base URL | 空（留空 = OpenAI 官方） |
| `wakeword` | 唤醒词 | `computer` |
| `wakeword_model_path` | 唤醒词模型（相对路径，自动拼接项目根目录） | `computer_v2.onnx` |
| `wakeword_threshold` | 唤醒词检测阈值 | `0.15` |
| `tts_voice` | 强制指定 TTS 音色（null = 自动选择） | `null` |
| `audio.silence_threshold` | 静音能量阈值 | `400.0` |
| `audio.silence_sec` | 静音结束录音秒数 | `1.5` |
| `audio.max_record_sec` | 最长录音秒数 | `25.0` |
| `audio.min_record_sec` | 最短录音秒数 | `0.5` |
| `memory.recent_keep` | 保留最近消息条数 | `6` |
| `memory.summary_trigger` | 触发摘要压缩的历史条数 | `12` |
| `memory.retrieve_top` | 关键词检索召回片段数 | `3` |

### 3.6 templates/（Web 前端）

- **chat.html**：星际迷航风格聊天界面，canvas 渲染舰载电脑 UI；支持多对话切换、新建对话、实时 AI 回复、工具日志、删除操作确认弹窗（CONFIRM/CANCEL）、SSE 实时事件流。
- **setup.html**：星际迷航风格设置界面，分模型（Model）、录音（Audio）、记忆（Memory）三个面板，通过 `/api/config/*` 接口读写配置。

### 3.7 音效与模型文件

| 文件 | 用途 |
|---|---|
| `wake_sound.wav` | 唤醒命中提示音（提示用户开始说话） |
| `complete.mp3` | 发送给 AI 时的音效 |
| `alert23.mp3` | AI 出错/掉线音效（第一段） |
| `commandfunctionsoffline_ep.mp3` | AI 出错/掉线音效（第二段） |
| `hey_computer.onnx` | 唤醒词模型（旧版） |
| `computer_v2.onnx` | 唤醒词模型（当前 config 使用） |

### 3.8 对话/（会话持久化目录）

每个对话一个 JSON 文件（`对话N.json`），包含 `name`、`summary`（摘要）、`history`（历史消息）、`updated_at`（最后活跃时间）。以磁盘文件为唯一真相，手动增删文件会立即反映到内存与 UI。

---

## 4. 运行说明

### 4.1 环境依赖

- **操作系统**：macOS（依赖 `afplay` 播放音频、`pynput` 监听键盘）。
- **Python 版本**：`>= 3.11`（推荐 3.11.9，项目虚拟环境 `~/agentscope-env` 即为此版本）。
- **依赖安装**（在项目目录下执行）：

```bash
pip install -r requirements.txt
# 国内网络建议使用清华镜像：
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

核心依赖：`numpy`、`sounddevice`、`faster-whisper`、`agentscope`、`flask`；可选依赖：`edge-tts`、`openwakeword`、`searchpin`。

### 4.2 配置准备

编辑 `config.json`，至少配置：

```json
{
  "api_key": "你的 API Key",
  "model": "你的模型名（如 gpt-4o / deepseek-chat）",
  "base_url": "OpenAI 兼容 Base URL（留空 = OpenAI 官方）"
}
```

也可通过 Web 设置界面（`http://127.0.0.1:7689`）配置。

### 4.3 启动方式

```bash
# 一键启动（语音助手 + Web 界面）
./start_all.sh
```

> 首次运行会自动下载 faster-whisper small 模型（通过 hf-mirror 镜像加速）。

### 4.4 使用方式

- **唤醒**：说出唤醒词（默认 `computer`），听到提示音后开始说话。
- **对话**：用自然语言下达指令，如「北京今天天气怎么样」「帮我找一下桌面上的合同」「运行 ls 命令」。
- **打断**：按 **Cmd + Shift + M** 打断当前 AI 回复/TTS 播放（需安装 `pynput`）。
- **对话管理**：语音说「创建对话」「切换到对话2」「列出对话」「清空对话」。
- **Web 界面**：浏览器打开 `http://127.0.0.1:7689`，`/chat` 聊天、`/setup` 设置。

### 4.5 常用命令行参数

```bash
# 列出音频输入设备
python voice_assistant.py --list-devices

# 列出全部 Edge TTS 可用音色
python voice_assistant.py --list-voices

# 指定模型/唤醒词/音色
python voice_assistant.py --model gpt-4o --wakeword computer --tts-voice zh-CN-XiaoxiaoNeural

# 开启调试日志
python voice_assistant.py --debug
```

### 4.6 安全机制

- **分级权限**：只读工具（Read/Glob/Grep/web_search/fetch_page）与写文件、移动、复制等命令直接放行；**仅删除类 Bash 命令**（`rm`/`rmdir`/`del`/`unlink`/`trash`/`find -delete` 等）需语音确认。
- **语音确认**：删除操作前 TTS 朗读操作描述，录音识别用户「确认/取消」。
- **Web 确认**：Web 界面中删除操作弹窗 CONFIRM/CANCEL，120 秒无操作默认拒绝。

---

## 5. 技术栈汇总

| 领域 | 技术 |
|---|---|
| 唤醒词 | openwakeword + onnxruntime（onnx 推理） |
| 语音识别 | faster-whisper（small 模型，CPU int8） |
| 语音合成 | edge-tts + afplay |
| Agent 框架 | agentscope（OpenAIChatModel + Agent + Toolkit + Middleware） |
| 音频采集 | sounddevice + numpy |
| Web 界面 | Flask + SSE（Server-Sent Events） |
| 桌面 GUI | tkinter |
| 联网搜索 | searchpin（多引擎并行 + 本地语义重排） |
| 持久化 | JSON 文件（对话/ 目录） |

---

## 6. 跨平台支持

本项目核心代码具备良好的跨平台基础，但仍有少量平台依赖需要改造才能完全跨平台运行。

### 6.1 已跨平台的部分

- **核心 Python 库均跨平台**：`sounddevice`（音频采集）、`faster-whisper`（语音识别）、`agentscope`（Agent 框架）、`flask`（Web 界面）、`edge-tts`（语音合成）、`openwakeword`（唤醒词）、`pynput`（键盘监听）均支持 Windows / macOS / Linux。
- **路径已相对化**：项目内所有路径均使用相对路径（基于 `PROJECT_ROOT` 动态拼接），无绝对路径硬编码，跨平台迁移时无需修改路径配置。

### 6.2 需改造的 3 处平台依赖

| # | 平台依赖 | 现状 | 跨平台改造方案 |
|---|---|---|---|
| 1 | **音频播放 `afplay`** | macOS 专属命令 | Windows / Linux 需替换为 `sounddevice` 直接播放，或改用 `pygame` / `playsound` 库播放 mp3 |
| 2 | **启动脚本 `start_all.sh`** | bash 脚本，仅限类 Unix 环境 | Windows 需提供 `.bat` 批处理脚本，或通过 Git Bash / WSL 运行 |
| 3 | **打断快捷键 `Cmd + Shift + M`** | macOS 专属（Cmd 键） | Windows 无 Cmd 键，需改为 `Win + Shift + M` 或 `Ctrl + Shift + M` |

---

## 7. 未来规划：硬件徽章 + 智能家居

### 7.1 最终目标

将本项目从「桌面语音助手」升级为**可随身携带的星际迷航徽章（combadge）**：佩戴徽章即可在任意位置通过语音与 AI 交互，并进一步实现**家具控制**——用自然语言指令操控家中智能设备。

### 7.2 硬件方案

- **主控**：树莓派 Zero 2W 或 ESP32-S3（低功耗、体积小，适合佩戴）。
- **外设**：麦克风（拾音）+ 扬声器（语音回复）+ 电池（供电）。
- **软件**：在硬件上运行本项目完整语音链路（唤醒 → 录音 → 转录 → LLM → TTS → 播放）。

### 7.3 任意位置 AI 交互

- 徽章通过 **Wi-Fi 接入局域网**，用户可在任意位置通过语音直接交互。
- 同时可通过浏览器访问 Web 界面（`http://127.0.0.1:7689`，局域网内为设备 IP:7689）进行可视化操作与配置。

### 7.4 家具控制（MQTT / Home Assistant）

- 通过 **MQTT / Home Assistant** 集成智能家居生态。
- AI agent 增加 **MQTT 工具**，将语音指令转化为设备控制命令。
- 交互链路示例：语音指令「打开客厅灯」→ agent 调用 MQTT 工具 → MQTT 发布控制消息 → 智能家居设备执行开灯。

*（内容由AI生成，仅供参考）*
