
# st-computer-lcars-ai-agent

Star Trek（星际迷航）DS9 LCARS 风格的本地 AI 语音助手，类似 OpenClaw。支持本地语音唤醒、语音识别（ASR）、大语言模型对话、TTS 朗读，并配有 Web 对话与管理界面（可在 Apple 风格与 LCARS 风格之间切换）。

> 项目在 Windows 上使用 Edge TTS 引擎提供高质量中文语音合成，自动识别所有可用中文声音并可动态切换。Web 服务、对话管理、工具调用等核心逻辑均为跨平台代码，在 macOS / Linux 上可完整运行 Web 界面与 AI 对话。

## 功能

| 模块 | 说明 |
|------|------|
| 唤醒词检测 | 说出 **"hey computer"** 唤醒，支持阈值自定义（默认 0.1） |
| 语音录制 | VAD 静音检测，自动切句，最短 0.5s / 最长 25s |
| 语音识别 | 本地 faster-whisper small 模型，CPU int8 量化 |
| AI 对话 | 接入任意兼容 OpenAI 格式的 LLM API，带上下文多轮对话 |
| 工具调用 | AI 可自主调用搜索引擎、读写文件、列出目录、执行 Shell、移动/删除文件 |
| TTS 朗读 | Edge TTS 引擎，自动识别所有中文声音，支持音色切换与语速调节 |
| 语音打断 | 朗读期间大声说话即可打断，自动切回聆听 |
| 多对话管理 | 创建 / 切换 / 删除对话，语音或 Web 均可操作 |
| Web 管理界面 | 浏览器查看对话历史、手动发送消息、管理对话与设置（含 LCARS / Apple 双主题） |

## 技术栈

```
┌─────────────────────────────────────────────────────┐
│  唤醒词          openwakeword + ONNX                │
│  语音识别        faster-whisper (small, CPU, int8)  │
│  LLM             OpenAI 兼容 API                    │
│  TTS             Edge TTS (edge-tts + pygame)          │
│  音频 I/O        pyaudio                            │
│  Web 后端        Flask                              │
│  Web 前端        HTML + 原生 JS (fetch 轮询)         │
│  搜索引擎        baidusearch / ddgs(DuckDuckGo) /    │
│                  自建 Bing 爬虫(BeautifulSoup)       │
│  文件操作        send2trash (删除到回收站)           │
│  运行环境        Windows 10/11 (完整功能)            │
│                  macOS / Linux (仅 Web + 对话，TTS 需 Edge TTS 联网)│
│  Python          3.9+                               │
└─────────────────────────────────────────────────────┘
```

## 程序运行流程

```
启动 app.py（Flask Web 服务，端口 8086）
         │
启动 voice_assistant.py（语音助手主循环，仅 Windows 完整可用）
         │
         ▼
┌─────────────────────────────────────┐
│  监听麦克风                          │
│  等待唤醒词 "hey computer"          │
│  （唤醒前低 CPU 占用）               │
└──────────────┬──────────────────────┘
               │ 检测到唤醒词
               ▼
┌─────────────────────────────────────┐
│  播放提示音 wake_sound.wav           │
│  录制音频至静音或超时                │
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
│  Edge TTS 朗读回复                     │
│  → 可被打断（大声说话触发）               │
│  → 语音命令可改语速/音色             │
└──────────────┬──────────────────────┘
               │
               ▼ 回到监听，等待下一次唤醒
```

## 目录结构

```
st-computer-lcars-ai-agent/
├── app.py                 # Flask Web 服务 + AI 工具调用后端
├── voice_assistant.py     # 语音助手主程序（唤醒/录音/ASR/TTS，仅 Windows 完整可用）
├── install_deps.py        # 依赖一键安装脚本（直接 pip install）
├── templates/
│   ├── index.html         # Web 管理界面（Apple + LCARS 双主题）
│   └── index_js_extracted.js  # 前端 JS 提取备份
├── data/                  # 运行时数据
│   ├── config.json        # 配置文件
│   ├── complete.wav       # AI 回复提示音
│   ├── wake_sound.wav     # 唤醒提示音
│   ├── error.wav          # 错误提示音
│   ├── command_code_verify.wav  # 密码验证提示音
│   ├── hey_computer.onnx  # 唤醒词模型（已附带）
│   ├── melspectrogram.onnx     # openwakeword 依赖模型（已附带）
│   ├── embedding_model.onnx    # openwakeword 依赖模型（已附带）
│   └── conversations/     # 对话历史 (JSON)
└── LICENSE
```

## 安装说明

### 1. 环境要求

- Python 3.9 及以上
- 麦克风和扬声器（语音功能需要，仅 Windows 生效）

### 2. 安装依赖

推荐使用项目自带的一键安装脚本，它会直接在当前 Python 环境中安装全部依赖：

```bash
python install_deps.py
```

脚本会安装以下跨平台依赖：

```
flask requests numpy pyaudio openwakeword faster-whisper
baidusearch ddgs beautifulsoup4 send2trash edge-tts pygame
```

> 说明：
> - 语音模型（`data/hey_computer.onnx`、`data/melspectrogram.onnx`、`data/embedding_model.onnx`）已随项目附带，无需联网下载。
> - `faster-whisper` 首次运行会自动下载 `small` 模型（约 1GB），已配置 `hf-mirror.com` 镜像加速。

手动安装（不使用脚本）：

```bash
pip install flask requests numpy pyaudio openwakeword faster-whisper \
            baidusearch ddgs beautifulsoup4 send2trash edge-tts pygame
```

### 3. 配置

编辑 `data/config.json`：

```json
{
    "home": "C:\\Users\\zhaoz",
    "paths": {
        "桌面": "D:\\DESK-P",
        "文档": "C:\\Users\\zhaoz\\Documents",
        "下载": "C:\\Users\\zhaoz\\Downloads",
        "项目": "D:\\DESK-P\\develop"
    },
    "security": {
        "password": "9472"
    },
    "api": {
        "base": "https://open.bigmodel.cn/api/paas/v4",
        "key": "sk-xxxxxxxx",
        "model": "glm-4"
    },
    "tts": {
        "voice": "zh-CN-XiaoyiNeural",
        "rate": 2,
        "processing_rate": 1
    },
    "wake": {
        "threshold": 0.1,
        "silence_threshold": 350,
        "silence_sec": 3.5,
        "stop_energy_threshold": 1200
    },
    "search": {
        "order": ["baidu", "bing", "ddgs"]
    }
}
```

| 字段 | 说明 |
|------|------|
| `home` | 用户主目录（AI 操作文件时的根路径） |
| `paths` | 路径别名映射（语音说"桌面"等简称时自动展开） |
| `security.password` | 删除文件时的安全密码（语音读出数字即可） |
| `api.base` | LLM API 地址（OpenAI 兼容格式，如 `.../v1` 或 `.../v4`） |
| `api.key` | API 密钥 |
| `api.model` | 模型名称 |
| `tts.voice` | 默认音色（Edge TTS ShortName，如 `zh-CN-XiaoyiNeural`），启动时自动识别所有可用中文声音 |
| `tts.rate` | 语速，兼容旧 SAPI -10~10 数值（自动映射为 Edge TTS 百分比），2≈+10% |
| `tts.processing_rate` | "处理中"提示音语速 |
| `wake.threshold` | 唤醒词灵敏度（越低越灵敏，默认 0.1） |
| `wake.silence_threshold` | 录音静音能量阈值（RMS） |
| `wake.silence_sec` | 静音多少秒后结束录音 |
| `wake.stop_energy_threshold` | 语音打断灵敏度（RMS 能量值） |
| `search.order` | 搜索引擎调用优先级（baidu / bing / ddgs） |

### 4. 启动

```bash
# 终端一：启动 Web 服务（端口 8086）
python app.py

# 终端二（Windows）：启动语音助手
python voice_assistant.py
```

启动后：
- 浏览器访问 `http://127.0.0.1:8086` 查看对话管理界面
- Windows 下对着麦克风说 **"hey computer"** 唤醒助手
- 听到提示音后说出你的问题，AI 回复会通过扬声器朗读
- 朗读期间大声说话可打断

> 在 macOS / Linux 上，仅运行 `python app.py` 即可使用 Web 对话界面；语音唤醒与 TTS 需要 Edge TTS 联网支持。

## 语音命令

唤醒后可直接用自然语言说出：

| 命令示例 | 功能 |
|----------|------|
| "帮我搜索一下今天天气" | AI 调用搜索引擎 |
| "读一下桌面上的 readme.txt" | AI 读取文件 |
| "把结果写入 result.txt" | AI 写入文件 |
| "语速快一点" | 调快 TTS 语速 |
| "换成晓晓的声音" | 切换 TTS 音色（晓晓/晓伊/云希/云健/云扬/云夏/小北/小妮） |
| "新建一个对话" | 创建新对话 |
| "切换到上一个对话" | 切换对话 |
| "删除当前对话" | 删除对话（需读出安全密码） |

## 许可证

见 [LICENSE](LICENSE)。

