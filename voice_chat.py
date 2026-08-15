# -*- coding: utf-8 -*-
"""
语音助手聊天界面 - AgentScope 2.x + Tkinter 聊天界面
不写死模型与 API 地址：支持任意 OpenAI 兼容端点（z.ai / DeepSeek / Kimi / OpenAI 等），
模型名与 Base URL 在界面中自由填写，Base URL 留空使用 OpenAI 官方端点。
运行:
    source ~/agentscope-env/bin/activate
    python voice_chat.py
"""
import asyncio
import os
import queue
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext

# 修复 tkinter 找不到 init.tcl（Marvis 嵌入式 Python 的 Tcl 在 runtime 目录）
def _fix_tcl_path():
    if "TCL_LIBRARY" in os.environ:
        return
    tcl_dir = os.path.expanduser(
        "~/Library/Application Support/com.tencent.mac.marvis/"
        "components/MarvisAgent/Versions/1.0.0.10225/runtime/"
        "python311/lib/tcl8.6"
    )
    if os.path.isfile(os.path.join(tcl_dir, "init.tcl")):
        os.environ["TCL_LIBRARY"] = tcl_dir
        tk_dir = os.path.dirname(tcl_dir) + "/tk8.6"
        if os.path.isdir(tk_dir):
            os.environ["TK_LIBRARY"] = tk_dir

_fix_tcl_path()

# Searchpin 模型下载走 hf-mirror 镜像；禁用 xet 协议避免直连被墙
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from agentscope.credential import OpenAICredential
from agentscope.model import OpenAIChatModel
from agentscope.agent import Agent
from agentscope.message import Msg, TextBlock
from agentscope.tool import (
    FunctionTool, Toolkit, ToolChunk,
    Read, Write, Bash, Glob, Grep,
)

DEFAULT_MODEL = ""  # 不写死模型，连接时由用户在界面填写任意 OpenAI 兼容模型名
DEFAULT_BASE_URL = ""  # 不写死地址，留空 = OpenAI 官方端点 https://api.openai.com/v1
DEFAULT_WORKDIR = os.path.expanduser("~/Desktop")  # Bash / 文件工具的默认工作目录
SYSTEM_PROMPT = (
    "你是一个理智高效的语音助手。回答简洁、准确、直击要点，不做多余寒暄。\n"
    "你有以下能力：\n"
    "1. web_search：多引擎并行联网搜索（Bing 国内/国际、搜狗、百度并行去重 + 本地语义重排），"
    "需要实时信息（天气、新闻、最新数据等）时主动调用；\n"
    "2. fetch_page：抓取指定 URL 的网页正文，用于深入阅读搜索结果中的页面；\n"
    "3. Read / Write / Glob / Grep：读写文件、搜索文件名、搜索文件内容；\n"
    "4. Bash：在 macOS 终端执行命令（文件操作、程序运行、系统命令等）。\n"
    "用户要求操作电脑（读文件、找文件、整理文件、运行命令等）时，先调用对应工具，再基于工具结果回答。"
)


# ---------- Searchpin 联网搜索（多引擎并行 + 本地重排） ----------

_searchpin_engine = None
_log_callback = None  # 工具执行日志回调，由 UI 注入，实时显示"正在做什么"


def _set_log_callback(cb):
    """注入工具日志回调，工具执行时实时推送进度到 UI"""
    global _log_callback
    _log_callback = cb


def _emit_log(text: str):
    if _log_callback is not None:
        _log_callback(text)


def _get_searchpin_engine():
    """Searchpin 引擎单例。首次加载 embedding 模型约 0.5-1s，之后复用。"""
    global _searchpin_engine
    if _searchpin_engine is None:
        from searchpin import SearchEngine
        _searchpin_engine = SearchEngine()
    return _searchpin_engine


def _web_search_impl(query: str, max_results: int = 5) -> ToolChunk:
    """联网搜索：Searchpin 多引擎并行（Bing/搜狗/百度）+ 本地语义重排"""
    _emit_log(f"正在联网搜索: {query}")
    try:
        engine = _get_searchpin_engine()
        raw = engine.search(query, max_results=max_results)
        results = raw.get("results") if isinstance(raw, dict) else raw
        lines = [
            f"{i}. {r.get('title', '')}\n   {r.get('url', '')}\n   {r.get('snippet', '')}"
            for i, r in enumerate(results[:max_results], 1)
        ]
        text = "\n".join(lines) if lines else "未找到相关结果"
        _emit_log(f"搜索完成: {len(lines)} 条结果")
        return ToolChunk(content=[TextBlock(text=text)])
    except Exception as e:
        _emit_log(f"搜索失败: {e}")
        return ToolChunk(content=[TextBlock(text=f"搜索失败: {e}")])


def _fetch_page_impl(url: str) -> ToolChunk:
    """抓取网页正文：输入完整 URL（http/https），返回提取后的页面文本"""
    _emit_log(f"正在抓取网页: {url}")
    try:
        engine = _get_searchpin_engine()
        page = engine.fetch(url)
        body = page.get("body", str(page)) if isinstance(page, dict) else str(page)
        body = body.strip()[:4000]
        _emit_log(f"网页抓取完成: {len(body)} 字符")
        return ToolChunk(content=[TextBlock(text=f"页面正文（{url}）:\n{body}")])
    except Exception as e:
        _emit_log(f"网页抓取失败: {e}")
        return ToolChunk(content=[TextBlock(text=f"抓取失败: {e}")])


# ---------- 带实时日志的本地工具包装（Read/Write/Bash/Glob/Grep） ----------

class LoggedRead(Read):
    async def call(self, file_path=None, **kwargs):
        _emit_log(f"正在读取文件: {file_path}")
        return await super().call(file_path=file_path, **kwargs)


class LoggedWrite(Write):
    async def call(self, file_path=None, content=None, **kwargs):
        _emit_log(f"正在写入文件: {file_path}（{len(content or '')} 字符）")
        return await super().call(file_path=file_path, content=content, **kwargs)


class LoggedBash(Bash):
    async def call(self, command=None, **kwargs):
        _emit_log(f"正在执行命令: {command}")
        async for chunk in super().call(command=command, **kwargs):
            yield chunk


class LoggedGlob(Glob):
    async def call(self, pattern=None, **kwargs):
        _emit_log(f"正在搜索文件: {pattern}")
        return await super().call(pattern=pattern, **kwargs)


class LoggedGrep(Grep):
    async def call(self, pattern=None, **kwargs):
        _emit_log(f"正在搜索文件内容: {pattern}")
        return await super().call(pattern=pattern, **kwargs)


def build_toolkit() -> Toolkit:
    """组装语音助手的工具集：多引擎联网搜索 + 网页抓取 + 文件操作 + 终端命令"""
    web_search = FunctionTool(
        func=_web_search_impl,
        name="web_search",
        description="多引擎并行联网搜索（Bing 国内/国际、搜狗、百度），返回标题、链接和摘要，已做去重与语义重排。输入关键词（如'北京 天气'、'AI 最新新闻'），用于查询天气、新闻、实时数据等需要联网的场景。",
        is_read_only=True,
    )
    fetch_page = FunctionTool(
        func=_fetch_page_impl,
        name="fetch_page",
        description="抓取指定 URL 的网页正文并提取可读文本。输入必须是完整 URL（以 http:// 或 https:// 开头），用于深入阅读搜索结果里的具体页面。",
        is_read_only=True,
    )
    file_tools = [
        LoggedRead(),
        LoggedWrite(),
        LoggedBash(cwd=DEFAULT_WORKDIR),
        LoggedGlob(),
        LoggedGrep(),
    ]
    return Toolkit(tools=[web_search, fetch_page, *file_tools])


class VoiceChatApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("语音助手")
        self.root.geometry("760x640")
        self.agent: Agent | None = None
        self.queue: queue.Queue = queue.Queue()
        self.pending_text: str = ""  # 未连接时暂存待发送消息

        self._build_config_bar()
        self._build_chat_area()
        self._build_input_area()

        self.root.after(100, self._poll_queue)

    # ---------- UI ----------
    def _build_config_bar(self):
        bar = ttk.Frame(self.root, padding=(8, 6))
        bar.pack(fill=tk.X)

        ttk.Label(bar, text="API Key:").grid(row=0, column=0, sticky=tk.W)
        self.key_var = tk.StringVar()
        ttk.Entry(bar, textvariable=self.key_var, width=38, show="*").grid(
            row=0, column=1, padx=6, pady=2)

        ttk.Label(bar, text="模型:").grid(row=1, column=0, sticky=tk.W)
        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        ttk.Entry(bar, textvariable=self.model_var, width=20).grid(
            row=1, column=1, sticky=tk.W, padx=6, pady=2)

        ttk.Label(bar, text="Base URL:").grid(row=1, column=2, sticky=tk.W)
        self.base_url_var = tk.StringVar(value=DEFAULT_BASE_URL)
        ttk.Entry(bar, textvariable=self.base_url_var, width=34).grid(
            row=1, column=3, sticky=tk.W, padx=6, pady=2)

        self.connect_btn = ttk.Button(bar, text="连接模型", command=self.connect)
        self.connect_btn.grid(row=0, column=4, rowspan=2, padx=8)

        self.status_var = tk.StringVar(value="未连接")
        ttk.Label(bar, textvariable=self.status_var, foreground="gray").grid(
            row=0, column=5, rowspan=2, padx=4)

        # 通用 OpenAI 兼容端点提示（灰色小字）
        ttk.Label(
            bar,
            text="支持任意 OpenAI 兼容端点：模型名任意填写（如 gpt-4o / deepseek-chat / kimi-k2）；"
                 "Base URL 留空使用 OpenAI 官方；API Key 留空时尝试读取环境变量 OPENAI_API_KEY",
            foreground="gray",
        ).grid(row=2, column=0, columnspan=6, sticky=tk.W, padx=2, pady=(4, 0))

    def _build_chat_area(self):
        frame = ttk.Frame(self.root, padding=(8, 0))
        frame.pack(fill=tk.BOTH, expand=True)
        self.chat = scrolledtext.ScrolledText(frame, wrap=tk.WORD,
                                              font=("PingFang SC", 13),
                                              state=tk.DISABLED)
        self.chat.pack(fill=tk.BOTH, expand=True)
        # 工具日志样式：灰色小字，实时显示 AI 正在执行的操作
        self.chat.tag_configure("tool_log", foreground="#888888",
                                font=("PingFang SC", 11))
        self.chat.tag_configure("user", foreground="#1a56db")
        self.chat.tag_configure("assistant", foreground="#111111")
        self.chat.tag_configure("system", foreground="#b3541e")

    def _build_input_area(self):
        frame = ttk.Frame(self.root, padding=(8, 6))
        frame.pack(fill=tk.X)
        self.msg_var = tk.StringVar()
        entry = ttk.Entry(frame, textvariable=self.msg_var, font=("PingFang SC", 13))
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        entry.bind("<Return>", lambda e: self.send())
        self.send_btn = ttk.Button(frame, text="发送", command=self.send)
        self.send_btn.pack(side=tk.RIGHT, padx=(8, 0))

    # ---------- 逻辑 ----------
    def connect(self):
        # API Key 留空时尝试读取环境变量（兼容 OPENAI_API_KEY 等标准用法）
        api_key = self.key_var.get().strip() or os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            self._append("【提示】请填写 API Key（或设置 OPENAI_API_KEY 环境变量）。", "system")
            return
        model_name = self.model_var.get().strip()
        if not model_name:
            self._append("【提示】请填写要使用的模型名，如 gpt-4o / deepseek-chat / kimi-k2。", "system")
            return
        base_url = self.base_url_var.get().strip() or None  # None = OpenAI 官方端点
        self.status_var.set("连接中...")
        self.connect_btn.config(state=tk.DISABLED)
        threading.Thread(target=self._connect_worker,
                         args=(api_key, model_name, base_url), daemon=True).start()

    def _connect_worker(self, api_key: str, model_name: str, base_url: str | None):
        try:
            credential = OpenAICredential(api_key=api_key, base_url=base_url)
            model = OpenAIChatModel(credential=credential, model=model_name)
            self.agent = Agent(name="voice_assistant", system_prompt=SYSTEM_PROMPT,
                               model=model, toolkit=build_toolkit())
            # 注入工具日志回调：工具执行时实时推送到 UI
            _set_log_callback(
                lambda text: self.queue.put(("tool_log", text)))
            self.queue.put(("connected", model_name))
        except Exception as e:
            self.queue.put(("error", f"连接失败: {e}"))

    def send(self):
        text = self.msg_var.get().strip()
        if not text:
            return
        self.msg_var.set("")
        self._append(f"你: {text}", "user")

        if self.agent is None:
            # 未连接：暂存消息，自动走连接流程，连接成功后自动发送
            self.pending_text = text
            self.connect()
            return

        self._append(f"助手: 正在思考...", "assistant")
        self.send_btn.config(state=tk.DISABLED)
        threading.Thread(target=self._send_worker, args=(text,), daemon=True).start()

    def _send_worker(self, text: str):
        try:
            msg = Msg(name="user", role="user", content=[TextBlock(text=text)])
            # Agent.reply 是异步方法，子线程内无事件循环，用 asyncio.run 跑
            reply = asyncio.run(self.agent.reply(msg))
            parts = [b.text for b in reply.content if b.type == "text"]
            answer = "\n".join(parts) if parts else "(模型无文本回复)"
            self.queue.put(("reply", answer))
        except Exception as e:
            self.queue.put(("error", f"请求失败: {e}"))

    # ---------- UI 更新 ----------
    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "connected":
                    self.status_var.set(f"已连接: {payload}")
                    self.send_btn.config(state=tk.NORMAL)
                    self.connect_btn.config(state=tk.NORMAL)
                    self._append("【语音助手已就绪，开始聊天】", "system")
                    # 连接成功后有暂存消息则自动发出
                    if self.pending_text:
                        text, self.pending_text = self.pending_text, ""
                        self._append(f"助手: 正在思考...", "assistant")
                        self.send_btn.config(state=tk.DISABLED)
                        threading.Thread(target=self._send_worker, args=(text,), daemon=True).start()
                elif kind == "reply":
                    self.send_btn.config(state=tk.NORMAL)
                    self._append(f"助手: {payload}", "assistant")
                elif kind == "tool_log":
                    # 实时工具日志：灰色小字显示 AI 正在执行的操作
                    self._append(f"  → {payload}", "tool_log")
                elif kind == "error":
                    self.status_var.set("未连接" if self.agent is None else "出错了")
                    self.send_btn.config(state=tk.NORMAL)
                    self.connect_btn.config(state=tk.NORMAL)
                    self._append(f"【{payload}】", "system")
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _append(self, text: str, kind: str):
        self.chat.config(state=tk.NORMAL)
        start = self.chat.index(tk.END)
        self.chat.insert(tk.END, text + "\n\n")
        end = self.chat.index(tk.END)
        if kind in ("user", "assistant", "system", "tool_log"):
            self.chat.tag_add(kind, start, end)
        self.chat.config(state=tk.DISABLED)
        self.chat.see(tk.END)


def main():
    root = tk.Tk()
    VoiceChatApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
