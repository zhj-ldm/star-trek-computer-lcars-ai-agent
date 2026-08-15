# -*- coding: utf-8 -*-
"""
voice_assistant.py - 语音助手（Star Trek 同款语音链路 + AgentScope agent 后端）

语音识别链路照搬 star-trek-computer-lcars-ai-agent，仅把 pyaudio 换成 sounddevice：
  - sounddevice 单流持续监听唤醒词（openwakeword / hey_computer.onnx / 阈值 0.15）
  - 唤醒后播放 wake_sound.wav 提示音（Star Trek 同款）
  - 录音：无条件录音 + RMS 能量阈值 VAD（静音 1.5s 结束），不依赖 webrtcvad
  - 转录：faster-whisper small / int8 / beam_size=5 / vad_filter=True / 语言自动检测

后端：AgentScope Agent 执行（联网搜索 / 本地文件操作），Edge TTS 朗读最终报告。

用法:
  export OPENAI_API_KEY=xxx
  python voice_assistant.py --model glm-4.7-flash --base-url https://open.bigmodel.cn/api/paas/v4
"""
import os
import sys

# Searchpin 模型下载走 hf-mirror 镜像；禁用 xet 协议避免直连被墙（须在 faster_whisper import 前）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import argparse
import asyncio
import json
import logging
import queue
import re
import subprocess
import threading
import time
import wave

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

from agentscope.credential import OpenAICredential
from agentscope.model import OpenAIChatModel
from agentscope.agent import Agent
from agentscope.message import Msg, TextBlock
from agentscope.middleware import MiddlewareBase
from agentscope.permission import PermissionBehavior, PermissionDecision

# 复用 voice_chat.py 的工具集与系统提示（不会启动 GUI）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import voice_chat
from voice_chat import build_toolkit, _set_log_callback

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Star Trek 同款音频参数 ──────────────────────
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.08
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)  # 1280
SILENCE_THRESHOLD = 400
SILENCE_SEC = 1.5
MAX_RECORD_SEC = 25
MIN_RECORD_SEC = 0.5
SILENCE_FRAMES = int(SILENCE_SEC / CHUNK_DURATION)
WAKE_THRESHOLD = 0.15
SENTENCE_END_PUNCTUATION = ['.', '?', '!', '\n', '。', '！', '？']

# 语音会话系统提示：保留语音助手工具能力，要求回复简短口语化，便于朗读
VOICE_SYSTEM_PROMPT = voice_chat.SYSTEM_PROMPT + (
    "\n5. 用户通过语音与你对话：回复必须简短、口语化、直击结论（1-3 句），"
    "不要输出长段落、列表或代码块；执行完操作后只汇报最终结果。\n"
    "6. 你是星际迷航中的ai电脑 ：语气冷静、专业、简洁，"
    "像星舰电脑一样直接给出结果，不做多余寒暄。\n"
    "7. 你收到的命令来自语音识别，可能存在识别错误；"
    "遇到明显不通顺或不合逻辑的指令时，结合上下文联想推断用户的真实意图，"
    "按推断后的意图执行。\n"
    "8. 当指令有歧义、无法理解或缺少关键信息（如\"那个文件\"指哪个文件）时，"
    "不要猜测执行，输出澄清问题，格式为 [CLARIFY]你的问题[/CLARIFY]，"
    "等待用户回答后再继续执行。\n"
    "9. 对话管理：用户可用语音直接管理对话，无需调用工具。"
    "创建新对话：说\"创建对话\"或\"新对话\"；"
    "切换对话：说\"切换对话 对话2\"或\"切换到对话2\"；"
    "列出对话：说\"有哪些对话\"或\"列出对话\"；"
    "清空当前对话：说\"清空对话\"。"
    "这些命令由系统直接处理并语音反馈结果。"
)


# ---------- TTS：Edge TTS（微软，多语言音色，可切换） ----------
DEFAULT_ZH_VOICE = "zh-CN-XiaoxiaoNeural"
DEFAULT_EN_VOICE = "en-US-JennyNeural"


def _pick_tts_voice(text):
    """按文本内容自动选择中/英文音色"""
    if re.search(r"[\u4e00-\u9fff]", text):
        return DEFAULT_ZH_VOICE
    return DEFAULT_EN_VOICE


async def _edge_tts_list_voices():
    import edge_tts
    return await edge_tts.list_voices()


def _clean_for_tts(text):
    """清洗 Markdown 符号后再朗读（显示原文不变）"""
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    t = re.sub(r"\*(.+?)\*", r"\1", t)
    t = re.sub(r"`(.+?)`", r"\1", t)
    t = re.sub(r"!\[.*?\]\(.*?\)", "", t)
    t = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", t)
    t = re.sub(r"^#{1,6}\s*", "", t, flags=re.MULTILINE)
    t = re.sub(r"^\s*[-*+]\s+", "", t, flags=re.MULTILINE)
    t = re.sub(r"\|.*?\|", "", t)
    t = re.sub(r"^---+$", "", t, flags=re.MULTILINE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


class EdgeTTSSynthesizer:
    """Edge TTS 合成 mp3 后用 afplay 播放（后台队列线程）"""

    def __init__(self, interrupt_event, voice_override=None):
        self.interrupt_event = interrupt_event
        self.voice_override = voice_override
        self.queue = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        import edge_tts
        while not self._stop.is_set():
            text = self.queue.get()
            if text is None:
                break
            try:
                voice = self.voice_override or _pick_tts_voice(text)
                clean = _clean_for_tts(text)
                if not clean:
                    continue
                mp3 = os.path.join(os.path.expanduser("~"), ".voice_tts_tmp.mp3")
                asyncio.run(edge_tts.Communicate(clean, voice).save(mp3))
                proc = subprocess.Popen(["afplay", mp3])
                while proc.poll() is None:
                    if self.interrupt_event.is_set():
                        proc.terminate()
                        break
                    time.sleep(0.05)
            except Exception as e:
                logging.warning(f"TTS 失败: {e}")
            finally:
                self.queue.task_done()

    def speak(self, text):
        self.queue.put(text)

    def clear_queue(self):
        """清空尚未播放的 TTS 队列（打断时调用）"""
        while True:
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except queue.Empty:
                break

    def stop(self):
        self._stop.set()
        self.queue.put(None)

    def close(self):
        self.stop()


# ---------- LLM：AgentScope agent（多会话 + 分层上下文管理） ----------
CLARIFY_RE = re.compile(r"\[CLARIFY\](.*?)\[/CLARIFY\]", re.S)

# 分层上下文参数：近期完整保留条数 / 触发摘要阈值 / 检索召回条数 / 检索片段长度
RECENT_KEEP = 6
SUMMARY_TRIGGER = 12
RETRIEVE_TOP = 3
RETRIEVE_CHARS = 600


def load_config():
    """加载项目根目录 config.json，返回 dict；文件缺失或解析失败返回 {}"""
    path = os.path.join(PROJECT_ROOT, "config.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"config.json 解析失败: {e}")
    return {}


def _apply_config_overrides():
    """启动时用 config.json 的 audio/memory 段覆盖默认常量（录音/记忆参数）"""
    cfg = load_config()
    global SILENCE_THRESHOLD, SILENCE_SEC, MAX_RECORD_SEC, MIN_RECORD_SEC
    global RECENT_KEEP, SUMMARY_TRIGGER, RETRIEVE_TOP, RETRIEVE_CHARS
    audio = cfg.get("audio") or {}
    if audio.get("silence_threshold") is not None:
        SILENCE_THRESHOLD = float(audio["silence_threshold"])
    if audio.get("silence_sec") is not None:
        SILENCE_SEC = float(audio["silence_sec"])
    if audio.get("max_record_sec") is not None:
        MAX_RECORD_SEC = float(audio["max_record_sec"])
    if audio.get("min_record_sec") is not None:
        MIN_RECORD_SEC = float(audio["min_record_sec"])
    memory = cfg.get("memory") or {}
    if memory.get("recent_keep") is not None:
        RECENT_KEEP = int(memory["recent_keep"])
    if memory.get("summary_trigger") is not None:
        SUMMARY_TRIGGER = int(memory["summary_trigger"])
    if memory.get("retrieve_top") is not None:
        RETRIEVE_TOP = int(memory["retrieve_top"])


_apply_config_overrides()

SUMMARY_PROMPT = (
    "请把以下对话历史压缩成一段简洁的中文摘要，保留：核心需求、项目背景、关键设定、"
    "重要结论与用户偏好；舍弃细碎寒暄与无关细节。直接输出摘要正文，不要任何前缀。\n\n"
    "对话历史：\n{history}"
)


class LLMHandler:
    """多会话 + 分层智能上下文管理：
    - 默认所有对话在同一个会话里，支持超长连续对话
    - 近期完整保留，久远历史自动摘要浓缩，提问时语义检索召回相关片段
    - 每次调用重建 agent，上下文长度严格受控，不把全部历史塞进模型窗口
    - 支持语音对话管理：创建/切换/列出/清空对话
    """

    def __init__(self, agent_factory, args, ask_user_callback=None, interrupt_event=None):
        self.args = args
        self._agent_factory = agent_factory
        self._ask_user = ask_user_callback
        self.interrupt_event = interrupt_event
        self.agent = None
        self._sessions = {}          # name -> {"history": [...], "summary": ""}
        self._current = None
        # 对话数据持久化目录：项目根目录/对话/
        self._chat_dir = os.path.join(PROJECT_ROOT, "对话")
        self._load_sessions()        # 加载已有对话；无对话时创建默认会话

    # ---------- 会话持久化 ----------
    def _load_chat_file(self, path):
        """从单个 JSON 文件加载一个会话；失败返回 None"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            fn = os.path.basename(path)
            name = data.get("name") or fn[:-5]
            return name, {
                "history": data.get("history", []),
                "summary": data.get("summary", ""),
                "updated_at": data.get("updated_at", 0),
            }
        except Exception as e:
            logging.warning(f"对话文件读取失败 {path}: {e}")
            return None

    def _save_current(self):
        """将当前会话写入 对话/ 目录（对话名.json）"""
        try:
            os.makedirs(self._chat_dir, exist_ok=True)
            data = {
                "name": self._current,
                "summary": self._cur()["summary"],
                "history": self._cur()["history"],
                "updated_at": time.time(),
            }
            self._cur()["updated_at"] = data["updated_at"]
            path = os.path.join(self._chat_dir, f"{self._current}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.warning(f"对话保存失败: {e}")

    def _resync_from_disk(self):
        """以磁盘文件为唯一真相：重新扫描 对话/ 目录。
        - 文件存在但内存没有 → 加载到内存
        - 文件比内存新   → 用磁盘版本覆盖内存
        - 内存有但文件没了 → 从内存中删除（用户在文件夹里删了 JSON）
        - 当前对话若已被删 → 切到第一个仍存在的对话，没有则新建
        """
        try:
            os.makedirs(self._chat_dir, exist_ok=True)
        except Exception as e:
            logging.warning(f"对话目录创建失败: {e}")
            return

        # Step 1: 收集磁盘上所有对话
        disk_sessions = {}
        try:
            for fn in sorted(os.listdir(self._chat_dir)):
                if not fn.endswith(".json"):
                    continue
                result = self._load_chat_file(os.path.join(self._chat_dir, fn))
                if result is None:
                    continue
                name, session = result
                disk_sessions[name] = session
        except Exception as e:
            logging.warning(f"对话目录扫描失败: {e}")

        # Step 2: 合并到内存 _sessions（磁盘优先）
        # 如果内存会话的 updated_at 比磁盘新（说明刚发送消息未被外部覆盖），保留内存
        for name, disk_sess in disk_sessions.items():
            mem_sess = self._sessions.get(name)
            if mem_sess is None:
                self._sessions[name] = disk_sess
            else:
                disk_ts = disk_sess.get("updated_at", 0)
                mem_ts = mem_sess.get("updated_at", 0)
                if disk_ts >= mem_ts:
                    self._sessions[name] = disk_sess

        # Step 3: 删除内存里已不存在于磁盘的会话
        for name in list(self._sessions.keys()):
            if name not in disk_sessions:
                del self._sessions[name]

        # Step 4: 如果当前对话被删，切到第一个仍存在的；都没有则新建一个
        if self._current and self._current not in self._sessions:
            if self._sessions:
                self._current = sorted(
                    self._sessions.keys(),
                    key=lambda n: self._sessions[n].get("updated_at", 0),
                    reverse=True,
                )[0]
            else:
                self._current = None

        if self._current is None:
            if self._sessions:
                # 有已有对话但当前未指定：默认使用第一个（对话1），不新建
                self._current = sorted(
                    self._sessions.keys(),
                    key=lambda n: self._sessions[n].get("updated_at", 0),
                    reverse=True,
                )[0]
            else:
                # 没有任何会话，创建默认（跳过磁盘重复扫描避免递归）
                self.new_chat(skip_resync=True)

    def _load_sessions(self):
        """启动时从 对话/ 目录加载全部已有对话；无对话则创建默认会话"""
        self._resync_from_disk()
        if self._sessions:
            try:
                self._rebuild_agent()
            except Exception as e:
                logging.warning(f"启动时 Agent 重建失败: {e}")
            logging.info(f"已加载 {len(self._sessions)} 个对话，当前: {self._current}")

    # ---------- Web 查询接口 ----------
    def get_chats_sorted(self):
        """返回按最后活跃时间倒序的对话名列表（最新对话在最上面）。
        每次调用前重新扫描磁盘，确保手动删除/添加的 JSON 立即反映到 UI。"""
        self._resync_from_disk()
        return sorted(self._sessions.keys(),
                      key=lambda n: self._sessions[n].get("updated_at", 0),
                      reverse=True)

    def get_history(self, name):
        """返回指定对话的历史消息列表；不存在返回 None"""
        self._resync_from_disk()
        s = self._sessions.get(name)
        return s.get("history", []) if s else None

    def get_summary(self, name):
        """返回指定对话的摘要；不存在返回 None"""
        self._resync_from_disk()
        s = self._sessions.get(name)
        return s.get("summary", "") if s else None

    # ---------- 会话管理 ----------
    def _cur(self):
        return self._sessions[self._current]

    def _next_chat_name(self):
        """根据磁盘上真实存在的 对话N.json 计算下一个名称。
        不会因为中间有被删的（如对话2没了）就复用，而是取最大 N+1。"""
        max_n = 0
        try:
            for fn in os.listdir(self._chat_dir):
                if not fn.endswith(".json"):
                    continue
                m = re.match(r"^对话(\d+)\.json$", fn)
                if m:
                    n = int(m.group(1))
                    if n > max_n:
                        max_n = n
        except Exception:
            pass
        # 如果文件夹里全是自定义命名（不在 对话N 格式），回退到内存计数
        if max_n == 0:
            max_n = len(self._sessions)
        return f"对话{max_n + 1}"

    def new_chat(self, skip_resync=False):
        # 先同步磁盘，否则在 len(_sessions) 与实际文件数不一致时会取到重名
        # 但如果是从 _resync_from_disk 内部调进来的，就不要再递归同步
        if not skip_resync:
            self._resync_from_disk()
        name = self._next_chat_name()
        self._sessions[name] = {"history": [], "summary": "", "updated_at": time.time()}
        self._current = name
        self._save_current()
        try:
            self._rebuild_agent()
        except Exception as e:
            logging.warning(f"Agent 重建失败（不影响会话保存）: {e}")
        logging.info(f"已创建并切换到新对话: {name}")
        return f"已创建新对话：{name}"

    def switch_chat(self, name):
        self._resync_from_disk()
        if name in self._sessions:
            self._current = name
            try:
                self._rebuild_agent()
            except Exception as e:
                logging.warning(f"切换对话时 Agent 重建失败: {e}")
            logging.info(f"已切换到对话: {name}")
            return f"已切换到对话：{name}"
        for k in self._sessions:
            if name in k:
                self._current = k
                try:
                    self._rebuild_agent()
                except Exception as e:
                    logging.warning(f"切换对话时 Agent 重建失败: {e}")
                logging.info(f"已切换到对话: {k}")
                return f"已切换到对话：{k}"
        return f"没有找到对话：{name}，当前对话：{self._current}"

    def list_chats(self):
        self._resync_from_disk()
        names = list(self._sessions.keys())
        cur = self._current
        lines = [f"{n}（当前）" if n == cur else n for n in names]
        return "当前对话列表：" + "、".join(lines)

    def reset_history(self):
        self._resync_from_disk()
        if self._current not in self._sessions:
            return "当前对话已不存在"
        self._cur()["history"] = []
        self._cur()["summary"] = ""
        try:
            self._rebuild_agent()
        except Exception as e:
            logging.warning(f"重置对话时 Agent 重建失败: {e}")
        self._save_current()
        logging.info("对话历史已清空")
        return "好的，当前对话已清空"

    def match_chat_command(self, user_text):
        """直接匹配用户语音中的对话管理命令（不经过 LLM），返回提示文本；非命令返回 None"""
        t = user_text.strip()
        self._resync_from_disk()
        if re.search(r"(创建对话|新建对话|新对话|开个新对话|新开对话)", t):
            return self.new_chat(skip_resync=True)
        if re.search(r"(切换对话|切换到|换到)", t):
            m = re.search(r"(?:切换对话|切换到|换到)[：: ]*([^\s，。！？]+)", t)
            if m:
                return self.switch_chat(m.group(1).strip())
            return "请告诉我要切换到哪个对话"
        if re.search(r"(列出对话|有哪些对话|对话列表|查看对话)", t):
            return self.list_chats()
        if re.search(r"(清空对话|重置对话|清空当前对话)", t):
            return self.reset_history()
        return None

    # ---------- 分层上下文 ----------
    def _recent_messages(self):
        return self._cur()["history"][-RECENT_KEEP:]

    def _retrieve(self, query):
        """轻量语义检索：按关键词重叠召回与当前问题最相关的历史片段"""
        q_tokens = set(re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z]{3,}", query.lower()))
        if not q_tokens:
            return ""
        scored = []
        for i, m in enumerate(self._cur()["history"]):
            c_tokens = set(re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z]{3,}", m["content"].lower()))
            overlap = len(q_tokens & c_tokens)
            if overlap > 0:
                scored.append((overlap, i, m["content"]))
        scored.sort(key=lambda x: (-x[0], x[1]))
        top = scored[:RETRIEVE_TOP]
        if not top:
            return ""
        return "\n".join(f"[历史片段{i + 1}] {c[:RETRIEVE_CHARS]}" for _, i, c in top)

    def _summarize(self, history_text):
        """用模型把久远历史压缩成摘要"""
        try:
            msg = Msg(name="user", role="user",
                      content=[TextBlock(text=SUMMARY_PROMPT.format(history=history_text))])
            reply = asyncio.run(self.agent.reply(msg))
            parts = [b.text for b in reply.content if b.type == "text"]
            return "\n".join(parts).strip()
        except Exception as e:
            logging.warning(f"摘要生成失败: {e}")
            return ""

    def _update_summary(self):
        """历史超阈值时：久远部分压缩进摘要，仅保留最近完整消息"""
        cur = self._cur()
        if len(cur["history"]) <= SUMMARY_TRIGGER:
            return
        keep = cur["history"][-RECENT_KEEP:]
        old = cur["history"][:-RECENT_KEEP]
        old_text = "\n".join(f"{m['role']}: {m['content']}" for m in old)
        new_summary = self._summarize(old_text)
        if new_summary:
            cur["summary"] = (cur["summary"] + "\n" + new_summary).strip()
        cur["history"] = keep
        logging.info(f"上下文已压缩：摘要 {len(cur['summary'])} 字，保留最近 {len(keep)} 条")

    def _rebuild_agent(self, user_text=None):
        """重建 agent：摘要 + 检索片段注入 system_prompt，近期完整对话作为消息传入"""
        extra = []
        s = self._cur()["summary"]
        if s:
            extra.append(f"【历史对话摘要（久远内容，简要参考）】\n{s}")
        if user_text:
            related = self._retrieve(user_text)
            if related:
                extra.append(f"【与当前问题相关的历史片段】\n{related}")
        sys_prompt = VOICE_SYSTEM_PROMPT
        if extra:
            sys_prompt += "\n\n" + "\n\n".join(extra)
        self.agent = self._agent_factory(system_prompt=sys_prompt)

    def _run_agent_reply(self, msgs):
        """运行 agent.reply，支持 interrupt_event 中途取消。

        原实现用 asyncio.run 同步阻塞，AI 生成/执行工具期间无法响应 Cmd + Shift + M 打断；
        这里改为独立事件循环 + task.cancel()，轮询打断标记，被设置则取消任务。
        返回 (reply, interrupted)。
        """
        if self.interrupt_event is None:
            return asyncio.run(self.agent.reply(msgs)), False

        async def _wait():
            task = asyncio.create_task(self.agent.reply(msgs))
            while not task.done():
                if self.interrupt_event.is_set():
                    task.cancel()
                    break
                await asyncio.sleep(0.05)
            try:
                return await task, False
            except asyncio.CancelledError:
                return None, True

        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_wait())
        finally:
            loop.close()

    # ---------- 主流程 ----------
    def chat_stream(self, user_text: str):
        """yield 最终回复文本；支持中途语音澄清与对话管理"""
        try:
            # 每次发消息前重新同步磁盘 —— 用户在文件夹里手动删了 JSON 要立即感知
            self._resync_from_disk()
            if self._current not in self._sessions:
                yield "当前对话已不存在，请重新选择或创建新对话"
                return
            self._rebuild_agent(user_text)
            msgs = [Msg(name=m["role"], role=m["role"], content=[TextBlock(text=m["content"])])
                    for m in self._recent_messages()]
            msgs.append(Msg(name="user", role="user", content=[TextBlock(text=user_text)]))
            for _ in range(3):  # 最多澄清 3 轮，防止死循环
                reply, interrupted = self._run_agent_reply(msgs)
                if interrupted:
                    logging.info("AI 回复已被 Cmd + Shift + M 打断（生成阶段）")
                    return
                parts = [b.text for b in reply.content if b.type == "text"]
                answer = "\n".join(parts).strip() if parts else ""
                if not answer:
                    logging.warning("模型无文本回复")
                    yield None
                    return

                # 对话管理命令（AI 输出触发）
                handled = self._handle_chat_command(answer)
                if handled is not None:
                    yield handled
                    return

                # 澄清标记：AI 需要用户补充信息
                m = CLARIFY_RE.search(answer)
                if m:
                    question = m.group(1).strip()
                    logging.info(f"[澄清] {question}")
                    user_answer = self._ask_user(question) if self._ask_user else None
                    if user_answer is None:
                        yield question
                        return
                    logging.info(f"[澄清回答] {user_answer}")
                    msgs = [Msg(name="user", role="user",
                                content=[TextBlock(text=f"（针对你刚才的澄清问题，我的回答是：{user_answer}）")])]
                    continue

                # 正常回复：记录历史 + 更新摘要 + 落盘
                self._cur()["history"].append({"role": "user", "content": user_text})
                self._cur()["history"].append({"role": "assistant", "content": answer})
                self._update_summary()
                self._save_current()
                logging.info(f"助手: {answer}")
                yield answer
                return
            yield None
        except Exception as e:
            logging.error(f"Agent 请求失败: {e}")
            yield None

    def _handle_chat_command(self, answer):
        """识别 AI 输出中的对话管理命令，返回提示文本；非命令返回 None"""
        a = answer.strip()
        if re.search(r"(创建对话|新建对话|新对话|开个新对话|新开对话)", a):
            return self.new_chat()
        if re.search(r"(切换对话|切换到|换到)", a):
            m = re.search(r"(?:切换对话|切换到|换到)[：: ]*([^\s，。！？]+)", a)
            if m:
                return self.switch_chat(m.group(1).strip())
            return "请告诉我要切换到哪个对话"
        if re.search(r"(列出对话|有哪些对话|对话列表|查看对话)", a):
            return self.list_chats()
        if re.search(r"(清空对话|重置对话|清空当前对话)", a):
            return self.reset_history()
        return None


# ---------- 语音助手主循环（Star Trek 同款） ----------
class VoiceAssistant:
    """Star Trek 同款语音链路：sounddevice 单流唤醒 + 能量 VAD 录音 + whisper small 转录 + AgentScope agent"""

    def __init__(self, args, agent_factory):
        self.args = args
        self.agent_factory = agent_factory
        self.interrupt_event = threading.Event()
        self.tts = EdgeTTSSynthesizer(
            self.interrupt_event,
            voice_override=getattr(args, "tts_voice", None),
        )
        self.llm = LLMHandler(self._create_agent, args, ask_user_callback=self._ask_user_via_voice,
                              interrupt_event=self.interrupt_event)

        # 唤醒词模型（onnx 推理）
        from openwakeword.model import Model
        if not os.path.exists(args.wakeword_model_path):
            raise FileNotFoundError(f"唤醒词模型不存在: {args.wakeword_model_path}")
        self.oww_model = Model(wakeword_models=[args.wakeword_model_path], inference_framework="onnx")
        self.wakeword_key = list(self.oww_model.models.keys())[0]
        logging.info("唤醒词模型就绪")

        # ASR：faster-whisper small（Star Trek 同款）
        logging.info("加载 faster-whisper small ...")
        self.asr = WhisperModel("small", device="cpu", compute_type="int8",
                                num_workers=2, cpu_threads=4)
        logging.info("ASR 就绪")

        # 唤醒提示音（Star Trek 同款）
        self.wake_sound = os.path.join(PROJECT_ROOT, "wake_sound.wav")

        # Cmd + Shift + M 打断：后台监听键盘，按下 Cmd + Shift + M 设置 interrupt_event
        self._key_listener = None
        self._start_key_listener()

    def _start_key_listener(self):
        """后台线程监听键盘：按下 Cmd + Shift + M 打断当前 AI 回复/TTS 播放"""
        try:
            from pynput import keyboard
        except ImportError:
            logging.warning("未安装 pynput，Cmd + Shift + M 打断功能不可用（pip install pynput）")
            return

        cmd_pressed = {"state": False}
        shift_pressed = {"state": False}

        def on_press(key):
            try:
                if key in (keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r):
                    cmd_pressed["state"] = True
                elif key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
                    shift_pressed["state"] = True
                elif cmd_pressed["state"] and shift_pressed["state"] and getattr(key, "char", None) in ("m", "M"):
                    self.interrupt_event.set()
                    logging.info("检测到 Cmd + Shift + M，打断 AI")
            except Exception:
                pass

        def on_release(key):
            try:
                if key in (keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r):
                    cmd_pressed["state"] = False
                elif key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
                    shift_pressed["state"] = False
            except Exception:
                pass

        self._key_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._key_listener.daemon = True
        self._key_listener.start()

    def _create_agent(self, system_prompt=None):
        """创建 agent 实例，绑定语音确认回调；system_prompt 可动态注入摘要/检索片段"""
        return self.agent_factory(confirm_callback=self._confirm_via_voice,
                                  system_prompt=system_prompt)

    def _play_wav(self, path):
        try:
            with wave.open(path, "rb") as wf:
                framerate = wf.getframerate()
                channels = wf.getnchannels()
                pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
            if channels > 1:
                pcm = pcm.reshape(-1, channels)
            sd.play(pcm, framerate)
            sd.wait()
        except Exception as e:
            logging.warning(f"提示音播放失败: {e}")

    def _play_mp3(self, path):
        """用 afplay 播放 mp3 音效"""
        try:
            subprocess.run(["afplay", path], check=False)
        except Exception as e:
            logging.warning(f"音效播放失败: {e}")

    def _record_until_silence(self, max_sec=MAX_RECORD_SEC, tag="录音"):
        """无条件录音 + RMS 能量 VAD（Star Trek 同款），静音 1.5s 结束，返回 bytes"""
        frames = []
        silence_count = 0
        has_speech = False
        start = time.time()
        logging.info(f"[{tag}] 正在录音…")
        try:
            with sd.InputStream(samplerate=SAMPLE_RATE, blocksize=CHUNK_SIZE,
                                channels=1, dtype="int16") as stream:
                while True:
                    data, _ = stream.read(CHUNK_SIZE)
                    frames.append(data)
                    audio = np.frombuffer(data, dtype=np.int16)
                    energy = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
                    elapsed = time.time() - start

                    if energy > SILENCE_THRESHOLD:
                        has_speech = True
                        silence_count = 0
                    else:
                        silence_count += 1

                    if has_speech and silence_count >= SILENCE_FRAMES and elapsed >= MIN_RECORD_SEC:
                        break
                    if elapsed >= max_sec:
                        break
        except Exception as e:
            logging.warning(f"[{tag}] {e}")
            return b""
        audio_bytes = b"".join(frames)
        logging.info(f"[{tag}] 时长 {len(audio_bytes) / (SAMPLE_RATE * 2):.1f}s")
        return audio_bytes

    def _transcribe(self, audio_bytes):
        """whisper small 转录（Star Trek 同款：beam_size=5 + vad_filter + 语言自动检测）"""
        if len(audio_bytes) < SAMPLE_RATE * 2 * MIN_RECORD_SEC:
            logging.info("[识别] 录音太短")
            return None
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        segments, info = self.asr.transcribe(
            audio_np, beam_size=5, language=None, vad_filter=True,
            vad_parameters=dict(threshold=0.5, min_silence_duration_ms=300))
        text = " ".join(s.text.strip() for s in segments).strip()
        return text if text else None

    def _confirm_via_voice(self, desc):
        """语音确认：TTS 朗读操作描述 + 提示音 + 短录音识别（确认/取消）"""
        self.tts.speak(f"我需要执行删除操作：{desc}，确认吗？")
        self.tts.queue.join()
        time.sleep(0.3)  # 等回声消散
        self._play_wav(self.wake_sound)  # 提示音，提示用户开始说话
        audio_bytes = self._record_until_silence(max_sec=8, tag="确认")
        text = self._transcribe(audio_bytes) if audio_bytes else None
        if text:
            logging.info(f"确认回答: {text}")
            lower = text.lower()
            if any(w in lower for w in ("确认", "执行", "可以", "好的", "确定", "同意", "继续", "yes", "ok", "对", "是")):
                return True
            if any(w in lower for w in ("取消", "不要", "不行", "拒绝", "算了", "别", "no", "stop", "不")):
                return False
        logging.info("确认回答无法识别，默认拒绝")
        return False

    def _ask_user_via_voice(self, question):
        """语音询问用户澄清问题，返回用户回答文本；无有效回答返回 None"""
        self.tts.speak(question)
        self.tts.queue.join()
        time.sleep(0.3)  # 等回声消散
        self._play_wav(self.wake_sound)  # 提示音，提示用户开始说话
        audio_bytes = self._record_until_silence(max_sec=8, tag="澄清")
        text = self._transcribe(audio_bytes) if audio_bytes else None
        if text:
            logging.info(f"澄清回答: {text}")
            return text
        logging.info("澄清无有效回答")
        return None

    def _handle_utterance(self, text):
        """AgentScope agent 执行 + Edge TTS 朗读最终报告"""
        sentences = re.split(r"(?<=[.?!。！？])\s+", text)
        text = sentences[0].strip()
        if not text:
            return
        logging.info(f"你（语音）: {text}")

        lower = text.lower()
        if any(w in lower for w in ("exit", "goodbye", "退出", "再见", "结束")):
            self.tts.speak("再见")
            self.tts.queue.join()
            sys.exit(0)

        # 对话管理快速路径（创建/切换/列出/清空对话，不经过 LLM）
        cmd = self.llm.match_chat_command(text)
        if cmd is not None:
            self.tts.speak(cmd)
            self.tts.queue.join()
            return

        sentence_buffer = ""
        got_reply = False
        self.interrupt_event.clear()
        self._play_mp3(os.path.join(PROJECT_ROOT, "complete.mp3"))
        for token in self.llm.chat_stream(text):
            if token is None or self.interrupt_event.is_set():
                break
            got_reply = True
            sentence_buffer += token
            if any(p in token for p in SENTENCE_END_PUNCTUATION):
                sentence = sentence_buffer.strip()
                if sentence:
                    self.tts.speak(sentence)
                sentence_buffer = ""
        if self.interrupt_event.is_set():
            # 用户按 Cmd + Shift + M 打断：停止剩余 TTS 播放，不再朗读
            self.tts.clear_queue()
            self.interrupt_event.clear()
            logging.info("AI 回复已被 Cmd + Shift + M 打断")
            return
        if sentence_buffer.strip():
            self.tts.speak(sentence_buffer.strip())
        self.tts.queue.join()
        if not got_reply:
            self._play_mp3(os.path.join(PROJECT_ROOT, "alert23.mp3"))
            self._play_mp3(os.path.join(PROJECT_ROOT, "commandfunctionsoffline_ep.mp3"))

    def run(self):
        """主循环：sounddevice 单流持续监听唤醒词，命中→关流→提示音→录音→转录→处理→重开流"""
        logging.info(f"就绪！说 '{self.args.wakeword}' 唤醒…")
        stream = sd.InputStream(samplerate=SAMPLE_RATE, blocksize=CHUNK_SIZE,
                                channels=1, dtype="int16")
        stream.start()
        try:
            while True:
                data, _ = stream.read(CHUNK_SIZE)
                audio = np.frombuffer(data, dtype=np.int16)
                prediction = self.oww_model.predict(audio)
                score = prediction.get(self.wakeword_key, 0.0)

                if score >= self.args.wakeword_threshold:
                    stream.stop()
                    stream.close()
                    logging.info(f"唤醒词命中 (score={score:.2f})")
                    self._play_wav(self.wake_sound)

                    audio_bytes = self._record_until_silence()
                    text = self._transcribe(audio_bytes) if audio_bytes else None
                    if text:
                        self._handle_utterance(text)
                    else:
                        logging.info("[识别] 无有效语音")

                    time.sleep(1.0)  # 等回声消散
                    self.oww_model.reset()
                    stream = sd.InputStream(samplerate=SAMPLE_RATE, blocksize=CHUNK_SIZE,
                                            channels=1, dtype="int16")
                    stream.start()
                    logging.info(f"就绪！说 '{self.args.wakeword}' 唤醒…")
        except KeyboardInterrupt:
            logging.info("停止中…")
        finally:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
            self.cleanup()

    def cleanup(self):
        self.tts.stop()
        if self._key_listener is not None:
            self._key_listener.stop()



# ---------- 分级权限：只读直接放行，仅删除类命令语音确认 ----------
DELETE_COMMAND_RE = re.compile(r"\b(rm|rmdir|del|unlink|trash)\b")


def _is_delete_command(command: str) -> bool:
    """判断 Bash 命令是否为删除类操作（rm/rmdir/del/unlink/trash/find -delete 等）"""
    if DELETE_COMMAND_RE.search(command):
        return True
    # find ... -delete / -exec rm 也是删除
    if re.search(r"\bfind\b.*-delete\b", command):
        return True
    if re.search(r"\bfind\b.*-exec\s+rm\b", command):
        return True
    return False


class VoicePermissionMiddleware(MiddlewareBase):
    """分级权限中间件：只读工具/只读命令直接放行；仅删除类 Bash 命令转语音确认，其余一律放行"""

    def __init__(self, confirm_callback):
        self._confirm = confirm_callback

    async def on_check_permission(self, agent, input_kwargs, next_handler):
        decision = await next_handler(**input_kwargs)
        if decision.behavior != PermissionBehavior.ASK:
            return decision  # ALLOW / DENY 直接透传

        tool = input_kwargs.get("tool")
        # 只读工具（Read/Glob/Grep/web_search/fetch_page）直接放行
        if getattr(tool, "is_read_only", False):
            return PermissionDecision(
                behavior=PermissionBehavior.ALLOW,
                message="只读工具自动放行",
            )

        tool_name = getattr(tool, "name", "?")
        tool_input = input_kwargs.get("tool_input", {})

        # 仅 Bash 删除类命令需要语音确认
        if tool_name == "Bash":
            command = (tool_input.get("command") or "").strip()
            if _is_delete_command(command):
                ok = await asyncio.to_thread(self._confirm, command)
                if ok:
                    return PermissionDecision(
                        behavior=PermissionBehavior.ALLOW,
                        message="用户语音确认",
                    )
                return PermissionDecision(
                    behavior=PermissionBehavior.DENY,
                    message="用户语音拒绝",
                )
            # 非删除命令（含写文件、移动、复制等）直接放行
            return PermissionDecision(
                behavior=PermissionBehavior.ALLOW,
                message="非删除命令自动放行",
            )

        # 其他工具（Write 等）直接放行
        return PermissionDecision(
            behavior=PermissionBehavior.ALLOW,
            message="非删除操作自动放行",
        )


# ---------- 入口 ----------
def build_agent(api_key: str, base_url: str | None, model: str, confirm_callback=None,
                system_prompt: str | None = None):
    credential = OpenAICredential(api_key=api_key, base_url=base_url)
    llm = OpenAIChatModel(credential=credential, model=model)
    middlewares = []
    if confirm_callback is not None:
        middlewares.append(VoicePermissionMiddleware(confirm_callback))
    return Agent(name="voice_assistant", system_prompt=system_prompt or VOICE_SYSTEM_PROMPT,
                 model=llm, toolkit=build_toolkit(), middlewares=middlewares)


def parse_args():
    p = argparse.ArgumentParser(description="语音助手（Star Trek 同款语音链路）")
    p.add_argument("--model", default=None, help="OpenAI 兼容模型名（如 glm-4.7-flash / gpt-4o / deepseek-chat）")
    p.add_argument("--base-url", default=None, help="OpenAI 兼容 Base URL，缺省用官方 https://api.openai.com/v1")
    p.add_argument("--api-key", default=None, help="API Key，缺省读 config.json 或环境变量 OPENAI_API_KEY")
    p.add_argument("--wakeword", default=None, help="唤醒词（默认 hey computer）")
    p.add_argument("--wakeword-model-path", default=None)
    p.add_argument("--wakeword-threshold", type=float, default=None,
                   help="唤醒词检测阈值（hey_computer 官方推荐 0.15）")
    p.add_argument("--tts-voice", default=None,
                   help="Edge TTS 音色（强制指定，如 zh-CN-XiaoxiaoNeural / en-US-JennyNeural）；"
                        "缺省按文本自动选择中/英文音色")
    p.add_argument("--list-voices", action="store_true", help="列出全部 Edge TTS 可用音色并退出")
    p.add_argument("--list-devices", action="store_true", help="列出音频输入设备并退出")
    p.add_argument("--debug", action="store_true", help="开启调试日志")
    return p.parse_args()


def make_voice_args(cli):
    """构造语音助手所需字段"""
    class A: pass
    a = A()
    a.wakeword = cli.wakeword
    a.wakeword_model_path = cli.wakeword_model_path
    a.wakeword_threshold = cli.wakeword_threshold
    a.tts_voice = cli.tts_voice
    return a


def main():
    cli = parse_args()
    logging.basicConfig(level=logging.DEBUG if cli.debug else logging.INFO,
                        format="%(levelname)s %(message)s")

    if cli.list_devices:
        print("\n--- 音频输入设备 ---")
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if dev.get("max_input_channels", 0) > 0:
                print(f"  Index {i}: {dev.get('name')}")
        return

    if cli.list_voices:
        async def _print_voices():
            voices = await asyncio.wait_for(_edge_tts_list_voices(), timeout=20)
            for v in voices:
                print(f"{v['ShortName']}\t{v['Locale']}\t{v['Gender']}\t{v['FriendlyName']}")
        asyncio.run(_print_voices())
        return

    # 配置优先级：命令行参数 > config.json > 默认值/环境变量
    cfg = load_config()
    model = cli.model or cfg.get("model")
    base_url = cli.base_url or cfg.get("base_url")
    api_key = cli.api_key or cfg.get("api_key") or os.environ.get("OPENAI_API_KEY", "").strip()
    wakeword = cli.wakeword or cfg.get("wakeword", "hey computer")
    wakeword_model_path = (cli.wakeword_model_path
                           or cfg.get("wakeword_model_path")
                           or os.path.join(PROJECT_ROOT, "hey_computer.onnx"))
    if wakeword_model_path and not os.path.isabs(wakeword_model_path):
        wakeword_model_path = os.path.join(PROJECT_ROOT, wakeword_model_path)
    wakeword_threshold = (cli.wakeword_threshold
                          if cli.wakeword_threshold is not None
                          else cfg.get("wakeword_threshold", WAKE_THRESHOLD))
    tts_voice = cli.tts_voice or cfg.get("tts_voice")

    if not model:
        print("缺少模型名：用 --model 传入或在 config.json 中配置 model")
        sys.exit(1)
    if not api_key:
        print("缺少 API Key：用 --api-key 传入、在 config.json 中配置 api_key，或设置环境变量 OPENAI_API_KEY")
        sys.exit(1)

    # 工具执行日志实时打印到终端（不会朗读）
    _set_log_callback(lambda text: logging.info(f"  → {text}"))

    logging.info(f"模型: {model} | Base URL: {base_url or '(OpenAI 官方)'} | 唤醒词: {wakeword}")
    agent_factory = lambda confirm_callback=None, system_prompt=None: build_agent(
        api_key, base_url, model, confirm_callback=confirm_callback,
        system_prompt=system_prompt)

    voice_args = make_voice_args(cli)
    voice_args.wakeword = wakeword
    voice_args.wakeword_model_path = wakeword_model_path
    voice_args.wakeword_threshold = wakeword_threshold
    voice_args.tts_voice = tts_voice
    assistant = VoiceAssistant(voice_args, agent_factory)
    assistant.run()


if __name__ == "__main__":
    main()
