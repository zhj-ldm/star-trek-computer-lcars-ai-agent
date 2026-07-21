
# st-computer-lcars-ai-agent

运行在Windows上，跨平台难度不大
star trek ds9 lcars风格 ，类似open claw的AI 语音助手，的支持本地语音识别、大语言模型对话、TTS 朗读，并配有常规 Web 对话和管理设置界面。可在设置切换lacrs UI和常规UI，本项目更侧重实用，但包含st风格
未来将要支持 ESP32 / Arduino 分布式拾音、扬声器节点，实现全屋任意位置唤醒 AI、内网家电设备控制。
目前打断方式没有找到合适的所以没有打断逻辑

## 功能

| 模块 | 说明 |
|------|------|
| 唤醒词检测 | 说出 **"hey computer"** 唤醒，支持阈值自定义 |
| 语音录制 | VAD 静音检测，自动切句，最短 0.5s / 最长 15s |
| 语音识别 | 本地 faster-whisper small 模型，CPU int8 量化 |
| AI 对话 | 接入任意兼容 OpenAI 格式的 LLM API，带上下文多轮对话 |
| 工具调用 | AI 可自主调用搜索引擎、读写文件、列出目录、执行 Shell |
| TTS 朗读 | Windows SAPI5 引擎，支持音色切换 (Huihui / Zira / David) 和语速调节 |
| 语音打断 | 朗读期间大声说话即可打断，自动切回聆听 |
| 多对话管理 | 创建 / 切换 / 删除对话，语音或 Web 均可操作 |
| Web 管理界面 | 浏览器查看对话历史、手动发送消息、管理对话 |

## 技术栈

```
┌─────────────────────────────────────────────────────┐
│  唤醒词          openwakeword + ONNX                │
│  语音识别        faster-whisper (small, CPU, int8)  │
│  LLM             OpenAI 兼容 API                    │
│  TTS             Windows SAPI5 (pywin32)            │
│  音频 I/O        pyaudio                            │
│  Web 后端        Flask                              │
│  Web 前端        HTML + 原生 JS (SSE 流式)          │
│  搜索引擎        DuckDuckGo (ddgs)                  │
│  运行环境        Windows 10/11, Python 3.9+         │
└─────────────────────────────────────────────────────┘
```

## 程序运行流程

```
启动 app.py（Web 服务，端口 8086）
         │
启动 voice_assistant.py（语音助手主循环）
         │
         ▼
┌─────────────────────────────────────┐
│  监听麦克风                          │
│  等待唤醒词 "hey_computer"          │
│  （唤醒前 ~3% CPU）                  │
└──────────────┬──────────────────────┘
               │ 检测到唤醒词
               ▼
┌─────────────────────────────────────┐
│  播放提示音 wake_sound.wav           │
│  录制音频至静音 1.5s 或超时 15s      │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  faster-whisper 转写为文本           │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  发送到 LLM API（带对话历史）        │
│  AI 可能调用工具（搜索/文件/Shell）  │
│  工具结果回传 AI，生成最终回复        │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  SAPI5 TTS 朗读回复                  │
│  → 可被打断（大声说话触发）           │
│  → 唤醒词工具调用可改语速/音色       │
└──────────────┬──────────────────────┘
               │
               ▼ 回到监听，等待下一次唤醒
```

## 目录结构

```
agnes-chat-backup/
├── app.py                 # Flask Web 服务 + AI 工具
├── voice_assistant.py     # 语音助手主程序
├── install_deps.py        # 依赖一键安装脚本
├── templates/
│   └── index.html         # Web 管理界面
└── data/                  # 运行时数据
    ├── config.json        # 配置文件
    ├── complete.wav       # AI 回复提示音
    ├── wake_sound.wav     # 唤醒提示音
    └── conversations/     # 对话历史 (JSON)
```

## 安装说明

### 1. 环境要求

- Windows 10 或 Windows 11
- Python 3.9 及以上
- 麦克风和扬声器

### 2. 安装依赖

```powershell
# 方式一：一键安装脚本
python install_deps.py
需要的几个onnx已重定向到data下，已附带

# 方式二：手动安装
pip install flask requests numpy pywin32 openwakeword pyaudio faster-whisper
```

> pyaudio 如果安装失败，通常是缺少 VC++ 运行时：
> 1. 下载安装 [VC++ Redist](https://aka.ms/vs/17/release/vc_redist.x64.exe)
> 2. 重试 `pip install pyaudio`
> 3. 或使用 `pip install pipwin && pipwin install pyaudio`

### 3. 配置

编辑 `data/config.json`：

```json
{
    "api": {
        "url": "https://your-api-endpoint/v1/chat/completions",
        "key": "sk-xxxxxxxx",
        "model": "gpt-4"
    },
    "tts": {
        "voice": "Microsoft Huihui Desktop",
        "rate": 3,
        "volume": 100,
        "processing_rate": 1
    },
    "wake": {
        "threshold": 0.15,
        "stop_energy_threshold": 1200
    }
}
```

| 字段 | 说明 |
|------|------|
| `api.url` | LLM API 地址（OpenAI 兼容格式） |
| `api.key` | API 密钥 |
| `api.model` | 模型名称 |
| `tts.voice` | 默认音色：`Huihui`(中文女) / `Zira`(英文女) / `David`(英文男) |
| `tts.rate` | 语速 -10~10，正值越快 |
| `tts.volume` | 音量 0~100 |
| `wake.threshold` | 唤醒词灵敏度（越低越灵敏） |
| `wake.stop_energy_threshold` | 语音打断灵敏度，RMS 能量值 |

### 4. 首次运行

首次启动时 faster-whisper 会自动下载 `small` 模型（约 1GB）。国内网络已配置 `hf-mirror.com` 镜像加速。

### 5. 启动

```powershell
# 终端一：启动 Web 服务
python app.py

# 终端二：启动语音助手
python voice_assistant.py
```

启动后：
- 浏览器访问 `http://127.0.0.1:5000` 查看对话管理界面
- 对着麦克风说 **"hey computer"** 唤醒助手
- 听到提示音后说出你的问题
- AI 回复会通过扬声器朗读
- 朗读期间大声说话可打断

## 语音命令

唤醒后可直接用自然语言说出：

| 命令示例 | 功能 |
|----------|------|
| "帮我搜索一下今天天气" | AI 调用搜索引擎 |
| "读一下桌面上的 readme.txt" | AI 读取文件 |
| "把结果写入 result.txt" | AI 写入文件 |
| "语速快一点" | 调快 TTS 语速 |
| "换成 David 的声音" | 切换 TTS 音色 |
| "新建一个对话" | 创建新对话 |
| "切换到上一个对话" | 切换对话 |
| "删除当前对话" | 删除对话 |

