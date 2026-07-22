"""
语音助手 — 唤醒词 "computer"
唤醒 → 录音 → ASR → AI 对话 → TTS 朗读
对话统一存入「语音对话」会话，Web 端实时可见
"""

import os
import sys
import time
import wave
import threading
import json
import traceback
import asyncio
import tempfile
import re

import numpy as np
import pyaudio
import requests
import edge_tts
import pygame

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
WAKE_SOUND = os.path.join(DATA_DIR, "wake_sound.wav")

# ── 音频参数 ──────────────────────────────────
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.08
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)  # 1280
FORMAT = pyaudio.paInt16
CHANNELS = 1

# ── VAD / 录音 ────────────────────────────────
SILENCE_THRESHOLD = 400
SILENCE_SEC = 1.5
MAX_RECORD_SEC = 25
MIN_RECORD_SEC = 0.5
SILENCE_FRAMES = int(SILENCE_SEC / CHUNK_DURATION)

# ── 唤醒词 ────────────────────────────────────
WAKE_WORD = "hey_computer"
WAKE_MODEL_PATH = os.path.join(DATA_DIR, "hey_computer.onnx")
WAKE_THRESHOLD = 0.15

# ── API ───────────────────────────────────────
API_BASE = "http://127.0.0.1:8086"
CONV_TITLE = "语音对话"
CONFIG_PATH = os.path.join(DATA_DIR, "config.json")


# ═══════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

def rms(audio):
    return float(np.sqrt(np.mean(np.square(audio.astype(np.float64)))))


def play_wav(path):
    try:
        wf = wave.open(path, 'rb')
        pa = pyaudio.PyAudio()
        st = pa.open(format=pa.get_format_from_width(wf.getsampwidth()),
                     channels=wf.getnchannels(), rate=wf.getframerate(),
                     output=True)
        chunk = wf.readframes(1024)
        while chunk:
            st.write(chunk)
            chunk = wf.readframes(1024)
        st.stop_stream(); st.close(); pa.terminate(); wf.close()
    except Exception:
        pass


# ═══════════════════════════════════════════════
#  Flask 后台
# ═══════════════════════════════════════════════

def _run_flask():
    sys.path.insert(0, BASE_DIR)
    from app import app
    app.run(host="127.0.0.1", port=8086, debug=False, use_reloader=False)


def wait_flask(timeout=10):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{API_BASE}/api/conversations", timeout=1)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.3)
    return False


# ═══════════════════════════════════════════════
#  对话管理
# ═══════════════════════════════════════════════

def get_or_create_voice_conv():
    """查找标题为「语音对话」的会话，不存在则创建"""
    r = requests.get(f"{API_BASE}/api/conversations")
    convs = r.json()
    for c in convs:
        if c.get("title") == CONV_TITLE:
            print(f"[会话] 复用已有「语音对话」: {c['id']}")
            return c["id"]
    r = requests.post(f"{API_BASE}/api/conversations",
                      json={"title": CONV_TITLE})
    conv = r.json()
    print(f"[会话] 新建「语音对话」: {conv['id']}")
    return conv["id"]


def get_conv_list():
    """获取所有对话列表"""
    r = requests.get(f"{API_BASE}/api/conversations")
    return r.json()


def get_conv_title(conv_id):
    """获取指定对话的标题"""
    convs = get_conv_list()
    for c in convs:
        if c["id"] == conv_id:
            return c.get("title", "未知")
    return "未知"


def create_conv(title):
    """创建新对话"""
    r = requests.post(f"{API_BASE}/api/conversations",
                      json={"title": title})
    return r.json()["id"]


# ── 语音命令关键词匹配 ────────────────────────

CREATE_KEYWORDS = ["新建对话", "创建对话", "新对话", "开个对话", "创建一个对话", "建立一个对话"]

SWITCH_KEYWORDS = ["切换到", "切换对话", "打开对话", "进入对话", "换到", "切换至"]

# ── 语速 / 音色命令 ──────────────────────────
SPEED_UP_KW    = ["快一点", "加速", "语速加快", "快点", "说话快点", "读快点"]
SPEED_DOWN_KW  = ["慢一点", "减速", "语速减慢", "慢点", "说话慢点", "读慢点"]
SPEED_RESET_KW = ["语速恢复", "语速默认", "正常语速", "默认语速"]
SPEED_SET_KW   = ["语速调到", "语速设为", "语速调到", "设置语速"]

VOICE_LIST_KW  = ["切换音色", "换声音", "音色列表", "有什么声音", "有哪些声音"]
VOICE_SET_KW   = ["切换音色到", "换成音色", "音色设为", "用声音", "切换为音色"]

DEFAULT_RATE = "+0%"

# Edge TTS 中文声音 ShortName → 显示名称映射（动态获取，此为备用）
FRIENDLY_NAMES = {
    "zh-CN-XiaoxiaoNeural":         "晓晓 (中文女声)",
    "zh-CN-XiaoyiNeural":           "晓伊 (中文女声)",
    "zh-CN-YunjianNeural":          "云健 (中文男声)",
    "zh-CN-YunxiNeural":            "云希 (中文男声)",
    "zh-CN-YunxiaNeural":           "云夏 (中文男声)",
    "zh-CN-YunyangNeural":          "云扬 (中文男声)",
    "zh-CN-liaoning-XiaobeiNeural": "小北 (东北话女声)",
    "zh-CN-shaanxi-XiaoniNeural":   "小妮 (陕西话女声)",
}


def match_command(text):
    """
    关键词匹配语音命令。
    返回 (cmd_type, payload)
      cmd_type: "create" / "switch" / "list" / "speed_up" / "speed_down" /
               "speed_reset" / "speed_set" / "voice_list" / "voice_set" / None
      payload: 对话标题 / 语速值 / 音色名 / None
    """
    text_stripped = text.strip()

    # ── 语速命令 ──
    for kw in SPEED_UP_KW:
        if kw in text_stripped:
            return ("speed_up", None)

    for kw in SPEED_DOWN_KW:
        if kw in text_stripped:
            return ("speed_down", None)

    for kw in SPEED_RESET_KW:
        if kw in text_stripped:
            return ("speed_reset", None)

    for kw in SPEED_SET_KW:
        if text_stripped.startswith(kw):
            val = text_stripped[len(kw):].strip()
            try:
                return ("speed_set", int(val))
            except ValueError:
                return ("speed_set", None)

    # ── 音色命令 ──
    for kw in VOICE_LIST_KW:
        if kw in text_stripped:
            return ("voice_list", None)

    for kw in VOICE_SET_KW:
        if text_stripped.startswith(kw):
            name = text_stripped[len(kw):].strip()
            if name:
                return ("voice_set", name)
            else:
                return ("voice_list", None)

    # 快捷音色切换 — Edge TTS 中文声音关键词映射
    name_map = {
        "晓晓": "zh-CN-XiaoxiaoNeural",
        "晓伊": "zh-CN-XiaoyiNeural",
        "云健": "zh-CN-YunjianNeural",
        "云希": "zh-CN-YunxiNeural",
        "云夏": "zh-CN-YunxiaNeural",
        "云扬": "zh-CN-YunyangNeural",
        "小北": "zh-CN-liaoning-XiaobeiNeural",
        "小妮": "zh-CN-shaanxi-XiaoniNeural",
        "男声": "zh-CN-YunjianNeural",
        "男音": "zh-CN-YunjianNeural",
        "女声": "zh-CN-XiaoxiaoNeural",
        "女音": "zh-CN-XiaoxiaoNeural",
    }
    for kw, vkey in name_map.items():
        if kw in text_stripped:
            return ("voice_set", vkey)

    # ── 对话管理命令 ──
    for kw in CREATE_KEYWORDS:
        if text_stripped == kw or text_stripped.startswith(kw):
            return ("create", None)

    for kw in SWITCH_KEYWORDS:
        if text_stripped.startswith(kw):
            title = text_stripped[len(kw):].strip()
            if title.endswith("对话"):
                title = title[:-2].strip()
            if title:
                return ("switch", title)
            else:
                return ("list", None)

    return (None, None)


def handle_command(cmd_type, payload, current_conv_id):
    """
    执行命令，返回 (new_conv_id, speak_text)。
    new_conv_id 为 None 表示无变化。
    """
    if cmd_type == "create":
        title = payload if payload else "未命名对话"
        new_id = create_conv(title)
        return (new_id, f"已创建新对话「{title}」")

    elif cmd_type == "switch":
        convs = get_conv_list()
        # 精确匹配
        for c in convs:
            if c.get("title") == payload:
                return (c["id"], f"已切换到「{payload}」")
        # 模糊匹配（标题包含关键词）
        matches = [c for c in convs if payload in c.get("title", "")]
        if len(matches) == 1:
            c = matches[0]
            return (c["id"], f"已切换到「{c['title']}」")
        elif len(matches) > 1:
            titles = "、".join(f"「{c['title']}」" for c in matches)
            return (None, f"找到多个匹配：{titles}，请说具体一点")
        else:
            return (None, f"未找到对话「{payload}」")

    elif cmd_type == "list":
        convs = get_conv_list()
        if not convs:
            return (None, "当前没有任何对话")
        titles = []
        for i, c in enumerate(convs, 1):
            marker = " ← 当前" if c["id"] == current_conv_id else ""
            titles.append(f"第{i}个：{c['title']}{marker}")
        return (None, "对话列表：" + "；".join(titles))

    return (None, None)


def _clean_for_tts(text):
    """去掉 Markdown 和特殊符号，只保留 TTS 可朗读的纯文字"""
    # 去掉图片语法 ![alt](url)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    # 去掉链接语法 [text](url)，保留链接文字
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # 去掉行内代码 `code`
    text = re.sub(r'`[^`]+`', '', text)
    # 去掉加粗/斜体标记 ** * ~~
    text = re.sub(r'\*{1,3}', '', text)
    text = re.sub(r'~~', '', text)
    # 去掉标题 # 标记（行首的 # 及空格）
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # 去掉无序列表标记 - * +（行首）
    text = re.sub(r'^[\-\*\+]\s+', '', text, flags=re.MULTILINE)
    # 去掉有序列表标记 1. 2. 等
    text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)
    # 去掉水平线 --- ***
    text = re.sub(r'^[\-\*]{3,}\s*$', '', text, flags=re.MULTILINE)
    # 去掉表格分隔符 |
    text = text.replace('|', ' ')
    # 去掉 HTML 标签 <...>
    text = re.sub(r'<[^>]+>', '', text)
    # 去掉反引号代码块标记 ```
    text = text.replace('```', '')
    # 多个连续空行压缩为单个换行
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 去掉行首行尾空白
    text = text.strip()
    return text

def send_to_ai(conv_id, text):
    """发送消息给 AI，返回 (回复文本, 工具步骤列表)"""
    r = requests.post(f"{API_BASE}/api/chat",
                      json={"message": text, "conversation_id": conv_id},
                      timeout=120)
    data = r.json()
    if data.get("error"):
        return f"错误: {data['error']}", []
    return data.get("reply", ""), data.get("tool_steps", [])


# ═══════════════════════════════════════════════
#  Edge TTS 工具函数
# ═══════════════════════════════════════════════

def _get_edge_voices_sync():
    """同步获取所有 Edge TTS 中文（zh-CN）声音列表。
    返回 [{"short_name": "zh-CN-XiaoxiaoNeural", "display_name": "晓晓 (中文女声)", "gender": "Female"}, ...]"""
    voices = []

    async def _fetch():
        all_voices = await edge_tts.list_voices()
        for v in all_voices:
            if v["Locale"].startswith("zh-CN"):
                sn = v["ShortName"]
                voices.append({
                    "short_name": sn,
                    "display_name": FRIENDLY_NAMES.get(sn, v.get("FriendlyName", sn)),
                    "gender": v["Gender"],
                })

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 在已运行的事件循环中
            import concurrent.futures
            future = concurrent.futures.Future()
            def _run():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                new_loop.run_until_complete(_fetch())
                new_loop.close()
                future.set_result(None)
            threading.Thread(target=_run, daemon=True).start()
            future.result(timeout=15)
        else:
            loop.run_until_complete(_fetch())
    except RuntimeError:
        asyncio.run(_fetch())
    except Exception:
        # 备用：返回内置列表
        for sn, name in FRIENDLY_NAMES.items():
            gender = "Female" if "Xiao" in sn and "Yun" not in sn else "Male"
            voices.append({"short_name": sn, "display_name": name, "gender": gender})

    return voices


def _sapi_rate_to_edge(rate):
    """将 SAPI 语速 (-10~10) 映射为 Edge TTS 百分比格式"""
    rate = max(-10, min(10, int(rate)))
    pct = int(rate * 5)  # -10→-50%, 0→0%, 10→+50%
    if pct >= 0:
        return f"+{pct}%"
    else:
        return f"{pct}%"


class VoiceAssistant:
    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self.oww = None
        self.asr = None
        self.tts = None
        self.running = True
        self.speaking = False       # TTS 朗读期间暂停唤醒词检测
        self.tts_active = False     # TTS 后台线程是否在播
        self.tts_stop_flag = False  # 语音打断标志

        # Edge TTS 声音列表
        self.edge_voices = []
        self.current_voice = "zh-CN-XiaoxiaoNeural"
        self.current_rate = "+0%"
        self.processing_rate = "+0%"

        # 从配置文件读取阈值
        try:
            cfg = load_config()
            w = cfg.get("wake", {})
            global WAKE_THRESHOLD, SILENCE_THRESHOLD, SILENCE_SEC, SILENCE_FRAMES
            WAKE_THRESHOLD = w.get("threshold", WAKE_THRESHOLD)
            SILENCE_THRESHOLD = w.get("silence_threshold", SILENCE_THRESHOLD)
            SILENCE_SEC = w.get("silence_sec", SILENCE_SEC)
            SILENCE_FRAMES = int(SILENCE_SEC / CHUNK_DURATION)
            print(f"[配置] 唤醒={WAKE_THRESHOLD}, 静音阈值={SILENCE_THRESHOLD}, 静音时长={SILENCE_SEC}s")
        except Exception:
            pass

    def load_models(self):
        print("[模型] 加载唤醒词模型 ...")
        from openwakeword.model import Model
        self.oww = Model(wakeword_models=[WAKE_MODEL_PATH], inference_framework="onnx")
        print("[模型] 唤醒词就绪")

        print("[模型] 加载 ASR 模型 faster-whisper small ...")
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        from faster_whisper import WhisperModel
        self.asr = WhisperModel("small", device="cpu", compute_type="int8",
                                num_workers=2, cpu_threads=4)
        print("[模型] ASR 就绪")

        print("[模型] 加载 TTS 引擎 (Edge TTS) ...")
        # 初始化 pygame mixer（用于播放 mp3）
        try:
            pygame.mixer.init()
        except Exception:
            pass

        # 获取所有 Edge TTS 中文声音
        self.edge_voices = _get_edge_voices_sync()
        if not self.edge_voices:
            # 兜底
            for sn, name in FRIENDLY_NAMES.items():
                gender = "Female" if "Xiao" in sn and "Yun" not in sn else "Male"
                self.edge_voices.append({"short_name": sn, "display_name": name, "gender": gender})
        print(f"[模型] Edge TTS 可用中文声音: {len(self.edge_voices)} 个")

        # 从配置文件读取 TTS 设置
        try:
            cfg = load_config()
            tts_cfg = cfg.get("tts", {})
            voice_setting = tts_cfg.get("voice", "zh-CN-XiaoxiaoNeural")
            rate_raw = tts_cfg.get("rate", 2)
            self.current_rate = _sapi_rate_to_edge(rate_raw)
            self.processing_rate = _sapi_rate_to_edge(tts_cfg.get("processing_rate", 1))
        except Exception:
            voice_setting = "zh-CN-XiaoxiaoNeural"
            self.current_rate = "+0%"
            self.processing_rate = "+0%"

        # 匹配当前音色
        self._set_voice(voice_setting)
        print(f"[模型] TTS 就绪 — {self.current_voice}, Rate={self.current_rate}")

    def speak(self, text):
        """后台异步朗读，不阻塞主循环（通过 self.tts_stop_flag 支持语音打断）"""
        text = _clean_for_tts(text)
        if not text or not text.strip():
            return

        # 极短纯符号跳过
        clean = text.strip().replace(" ", "").replace("\n", "")
        if len(clean) <= 1 and not any('\u4e00' <= c <= '\u9fff' for c in clean):
            return

        self.tts_stop_flag = False
        self.tts_active = True

        voice = self.current_voice
        rate = self.current_rate

        def _tts_thread():
            tmp_path = None
            stopped = False
            try:
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
                    tmp_path = f.name

                # 异步合成
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                communicate = edge_tts.Communicate(text, voice, rate=rate)
                loop.run_until_complete(communicate.save(tmp_path))
                loop.close()

                # 播放 mp3（支持打断）
                pygame.mixer.music.load(tmp_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    if self.tts_stop_flag:
                        pygame.mixer.music.stop()
                        stopped = True
                        print("[TTS] 已被语音打断")
                        break
                    time.sleep(0.05)
            except Exception as e:
                print(f"[TTS错误] {e}")
            finally:
                self.tts_active = False
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
                if not stopped:
                    time.sleep(0.5)  # 正常播完后回声消散

        self.tts_thread = threading.Thread(target=_tts_thread, daemon=True)
        self.tts_thread.start()

    def wait_speak(self):
        """阻塞等待后台 TTS 完成（用于短命令等不需要打断的场景）"""
        if hasattr(self, 'tts_thread') and self.tts_thread and self.tts_thread.is_alive():
            self.tts_thread.join(timeout=30)
        self.tts_active = False

    def _speak_processing(self):
        """以独立语速朗读 Processing 提示，不随 AI 回复语速变化"""
        saved_rate = self.current_rate
        try:
            self.current_rate = self.processing_rate
            # 用同步方式朗读短文本
            self.speak("Processing")
            self.wait_speak()
        except Exception as e:
            print(f"[TTS错误] {e}")
        finally:
            self.current_rate = saved_rate

    def record_until_silence(self):
        """
        唤醒后录音，VAD 静音检测。
        返回 (audio_bytes, sample_width)
        """
        stream = self.pa.open(format=FORMAT, channels=CHANNELS,
                              rate=SAMPLE_RATE, input=True,
                              frames_per_buffer=CHUNK_SIZE)

        frames = []
        silence_count = 0
        has_speech = False
        record_start = time.time()
        print("[录音] 开始 ...")

        while self.running:
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            frames.append(data)
            audio = np.frombuffer(data, dtype=np.int16)
            energy = rms(audio)

            elapsed = time.time() - record_start

            if energy > SILENCE_THRESHOLD:
                has_speech = True
                silence_count = 0
            else:
                silence_count += 1

            # 静音超时且已检测到语音且超过最短录音 → 结束
            if has_speech and silence_count >= SILENCE_FRAMES and elapsed >= MIN_RECORD_SEC:
                break

            # 超长兜底
            if elapsed >= MAX_RECORD_SEC:
                break

        stream.stop_stream()
        stream.close()

        audio_bytes = b''.join(frames)
        duration = len(audio_bytes) / (SAMPLE_RATE * 2)
        print(f"[录音] 结束，时长 {duration:.1f}s")
        return audio_bytes

    def transcribe(self, audio_bytes):
        """Whisper 转写音频，返回文本或 None"""
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        segments, info = self.asr.transcribe(audio_np, beam_size=5,
                                              language=None,
                                              vad_filter=True,
                                              vad_parameters=dict(
                                                  threshold=0.5,
                                                  min_silence_duration_ms=300
                                              ))
        text = " ".join(s.text.strip() for s in segments).strip()
        return text if text else None

    def _set_voice(self, voice_id):
        """切换音色（Edge TTS ShortName），返回描述文本"""
        voice_id = voice_id.strip()
        # 直接在 edge_voices 中匹配
        for v in self.edge_voices:
            if v["short_name"] == voice_id:
                self.current_voice = voice_id
                return f"已切换到{v['display_name']}"

        # 兼容旧格式：尝试在 FRIENDLY_NAMES 中匹配
        if voice_id in FRIENDLY_NAMES:
            sn = voice_id
            display = FRIENDLY_NAMES[sn]
            self.current_voice = sn
            return f"已切换到{display}"

        # 显示名模糊匹配
        for v in self.edge_voices:
            if voice_id in v["display_name"]:
                self.current_voice = v["short_name"]
                return f"已切换到{v['display_name']}"

        # 未找到
        available = "、".join(v["display_name"] for v in self.edge_voices[:5])
        return f"未找到音色「{voice_id}」。可用：{available}等"

    def _set_rate(self, rate_val):
        """设置语速（Edge TTS 百分比格式或 SAPI -10~10 兼容），返回描述文本"""
        if isinstance(rate_val, str) and "%" in rate_val:
            self.current_rate = rate_val
            return f"语速已设为 {rate_val}"
        # 兼容旧的数字输入
        rate_int = max(-10, min(10, int(rate_val)))
        edge_rate = _sapi_rate_to_edge(rate_int)
        self.current_rate = edge_rate
        return f"语速已设为 {edge_rate}（原值 {rate_int}）"

    def _handle_tts_command(self, cmd_type, payload):
        """处理语速/音色命令，返回 (speak_text, is_command)"""
        if cmd_type == "speed_up":
            # 提升 10%
            current_pct = int(self.current_rate.replace("%", "").replace("+", ""))
            new_pct = current_pct + 10
            new_rate = f"+{new_pct}%"
            return (self._set_rate(new_rate), True)

        elif cmd_type == "speed_down":
            # 降低 10%
            current_pct = int(self.current_rate.replace("%", "").replace("+", ""))
            new_pct = max(-50, current_pct - 10)
            if new_pct >= 0:
                new_rate = f"+{new_pct}%"
            else:
                new_rate = f"{new_pct}%"
            return (self._set_rate(new_rate), True)

        elif cmd_type == "speed_reset":
            return (self._set_rate(DEFAULT_RATE), True)

        elif cmd_type == "speed_set":
            if payload is not None:
                return (self._set_rate(payload), True)
            else:
                return ("请说出语速数值，如：语速调到5", True)

        elif cmd_type == "voice_list":
            lines = ["可用音色："]
            for v in self.edge_voices:
                marker = ""
                if v["short_name"] == self.current_voice:
                    marker = " ← 当前"
                lines.append(f"  {v['display_name']}{marker}")
            return ("\n".join(lines), True)

        elif cmd_type == "voice_set":
            return (self._set_voice(payload), True)

        return (None, False)

    def _reload_tts_from_config(self):
        """从 config.json 重新加载并应用 TTS 设置（语音/语速）"""
        try:
            cfg = load_config()
            tts_cfg = cfg.get("tts", {})
            voice_setting = tts_cfg.get("voice", "zh-CN-XiaoxiaoNeural")
            rate_raw = tts_cfg.get("rate", 2)
            self.current_rate = _sapi_rate_to_edge(rate_raw)
            self.processing_rate = _sapi_rate_to_edge(tts_cfg.get("processing_rate", 1))

            # 尝试匹配
            found = False
            for v in self.edge_voices:
                if v["short_name"] == voice_setting:
                    self.current_voice = voice_setting
                    found = True
                    break

            if not found:
                # 兼容旧格式
                if voice_setting in FRIENDLY_NAMES:
                    self.current_voice = voice_setting
                else:
                    # 尝试显示名模糊匹配
                    for v in self.edge_voices:
                        if voice_setting in v["display_name"]:
                            self.current_voice = v["short_name"]
                            found = True
                            break
                    if not found:
                        print(f"[TTS] 未找到音色 '{voice_setting}'，保持当前")
        except Exception:
            pass

    def _apply_pending_actions(self, conv_id):
        """读取 pending_action.json，执行 TTS 配置重载和对话切换，返回 (new_conv_id, note)"""
        pending_file = os.path.join(DATA_DIR, "pending_action.json")
        if not os.path.exists(pending_file):
            return (None, "")

        try:
            with open(pending_file, "r") as f:
                actions = json.load(f)
        except Exception:
            return (None, "")

        # 删除文件防止重复执行
        try:
            os.remove(pending_file)
        except Exception:
            pass

        notes = []
        new_conv_id = None

        for action in actions:
            atype = action.get("type", "")
            if atype in ("set_rate", "set_voice"):
                self._reload_tts_from_config()
                if atype == "set_rate":
                    notes.append(f"语速→{action.get('value')}")
                else:
                    notes.append(f"音色→{action.get('value')}")
            elif atype in ("new_conv", "switch_conv"):
                new_conv_id = action.get("conv_id")
                notes.append(f"对话→{action.get('title', new_conv_id)}")
            elif atype == "delete_conv":
                # 如果当前对话被删除，切到最新对话
                if conv_id == action.get("conv_id"):
                    convs = get_conv_list()
                    new_conv_id = convs[0]["id"] if convs else create_conv("新对话")
                    notes.append(f"当前对话已删除，切换→{new_conv_id}")
                else:
                    notes.append(f"删除对话「{action.get('title')}」")

        return (new_conv_id, "; ".join(notes) if notes else "")

    def _reopen_stream(self):
        """关闭旧 stream 并打开新 stream，回归初始监听状态"""
        time.sleep(2.0)  # 等待 TTS 回声充分消散
        if self.oww is not None:
            self.oww.reset()  # 重置 OWW 内部状态，彻底回到初始
        return self.pa.open(format=FORMAT, channels=CHANNELS,
                            rate=SAMPLE_RATE, input=True,
                            frames_per_buffer=CHUNK_SIZE)

    def listen_loop(self, conv_id):
        """主循环：持续监听唤醒词"""
        print(f"[监听] 进入监听模式，说 '{WAKE_WORD}' 唤醒 ...")
        stream = self.pa.open(format=FORMAT, channels=CHANNELS,
                              rate=SAMPLE_RATE, input=True,
                              frames_per_buffer=CHUNK_SIZE)
        self.speaking = False

        while self.running:
            try:
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)

                if self.speaking:
                    if not self.tts_active:
                        # TTS 播放完毕 → 回到监听
                        print("[完成] 本轮结束，回到监听\n")
                        if self.oww is not None:
                            self.oww.reset()
                        stream = self.pa.open(format=FORMAT, channels=CHANNELS,
                                              rate=SAMPLE_RATE, input=True,
                                              frames_per_buffer=CHUNK_SIZE)
                        self.speaking = False
                    continue

                audio = np.frombuffer(data, dtype=np.int16)
                pred = self.oww.predict(audio)
                score = pred.get(WAKE_WORD, 0.0)

                if score >= WAKE_THRESHOLD:
                    stream.stop_stream()
                    stream.close()

                    print(f"[唤醒] {WAKE_WORD} ({score:.2f})")
                    play_wav(WAKE_SOUND)

                    # 录音
                    audio_bytes = self.record_until_silence()
                    if len(audio_bytes) < SAMPLE_RATE * 2 * MIN_RECORD_SEC:
                        print("[跳过] 录音太短")
                        stream = self._reopen_stream()
                        continue

                    # ASR
                    print("[识别] 转写中 ...")
                    text = self.transcribe(audio_bytes)
                    if not text:
                        print("[识别] 无有效语音")
                        stream = self._reopen_stream()
                        continue

                    print(f"[识别] {text}")

                    # ── 语音命令匹配 ──
                    cmd_type, payload = match_command(text)
                    if cmd_type:
                        print(f"[命令] {cmd_type} {payload or ''}")

                        # 先检查 TTS 控制命令（语速 / 音色）
                        speak_text, is_cmd = self._handle_tts_command(cmd_type, payload)
                        if is_cmd:
                            self.speaking = True
                            self.speak(speak_text)
                            self.wait_speak()
                            print("[完成] TTS 命令执行完毕，回到监听\n")
                            stream = self._reopen_stream()
                            self.speaking = False
                            continue

                        # 对话管理命令
                        new_id, speak_text = handle_command(cmd_type, payload, conv_id)
                        self.speaking = True
                        self.speak(speak_text)
                        self.wait_speak()
                        if new_id:
                            conv_id = new_id
                            print(f"[会话] 已切换到: {conv_id} ({get_conv_title(conv_id)})")
                        print("[完成] 命令执行完毕，回到监听\n")
                        stream = self._reopen_stream()
                        self.speaking = False
                        continue

                    # 播放提示音 → 发送 AI → TTS 朗读回复（后台异步 + 语音打断）
                    print("[AI] 发送中 ...")
                    self.speaking = True
                    play_wav(os.path.join(DATA_DIR, "complete.wav"))
                    reply, tool_steps = send_to_ai(conv_id, text)

                    # 处理 AI 工具调用产生的副作用（TTS 配置变更 / 对话切换）
                    new_cid, action_note = self._apply_pending_actions(conv_id)
                    if new_cid:
                        conv_id = new_cid
                        print(f"[会话] AI 已切换到: {conv_id} ({get_conv_title(conv_id)})")
                    if action_note:
                        print(f"[动作] {action_note}")

                    for step in tool_steps:
                        print(f"[步骤] {step}")
                        # 密码验证音效：步骤中检测"PASSWORD_REQUIRED"→错误
                        if "PASSWORD_REQUIRED" in step:
                            play_wav(os.path.join(DATA_DIR, "error.wav"))
                    # 密码正确：从 AI 回复中检测"密码正确"→验证音效
                    if "密码正确" in reply:
                        play_wav(os.path.join(DATA_DIR, "command_code_verify.wav"))
                    print(f"[AI] {reply}")
                    self._stop_energy_frames = 0
                    self.tts_stop_flag = False
                    self.speak(reply)  # 异步，立即返回

                    # 立即打开流（无延迟），主循环 speaking 分支接管打断检测
                    stream = self.pa.open(format=FORMAT, channels=CHANNELS,
                                          rate=SAMPLE_RATE, input=True,
                                          frames_per_buffer=CHUNK_SIZE)
                    # 注意：self.speaking 保持 True，由主循环判断 TTS 是否结束

            except OSError as e:
                # 音频设备异常，自动重建 stream，失败则循环重试
                errno = getattr(e, 'errno', 'N/A')
                print(f"[音频] 设备异常 (errno={errno})，重建音频流 ...")
                traceback.print_exc()
                try:
                    stream.close()
                except Exception:
                    pass
                for attempt in range(1, 11):
                    try:
                        time.sleep(2)
                        stream = self.pa.open(format=FORMAT, channels=CHANNELS,
                                              rate=SAMPLE_RATE, input=True,
                                              frames_per_buffer=CHUNK_SIZE)
                        if self.oww is not None:
                            self.oww.reset()
                        print(f"[音频] 音频流已重建（第 {attempt} 次尝试）")
                        break
                    except Exception as e2:
                        print(f"[音频] 重建失败（第 {attempt} 次）: {e2}")
                        if attempt == 10:
                            print("[音频] 重建 10 次均失败，退出监听")
                            self.running = False
            except Exception as e:
                traceback.print_exc()
                time.sleep(0.5)

        stream.stop_stream()
        stream.close()

    def cleanup(self):
        self.pa.terminate()


# ═══════════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 50)
    print(f"  Voice Assistant — 唤醒词: {WAKE_WORD}")
    print("=" * 50)

    # 1. 启动 Flask
    print("[启动] 后台启动 Web 服务 ...")
    threading.Thread(target=_run_flask, daemon=True).start()

    if not wait_flask():
        print("[错误] Flask 启动超时")
        sys.exit(1)
    print("[启动] Web 服务就绪 (http://127.0.0.1:8086)")

    # 2. 获取/创建语音对话
    conv_id = get_or_create_voice_conv()

    # 3. 初始化语音助手
    va = VoiceAssistant()
    try:
        va.load_models()
        print("\n" + "=" * 50)
        print(f"  一切就绪，说 '{WAKE_WORD}' 开始对话")
        print("  按 Ctrl+C 退出")
        print("=" * 50 + "\n")
        va.listen_loop(conv_id)
    except KeyboardInterrupt:
        print("\n[退出] 再见")
    finally:
        va.cleanup()
