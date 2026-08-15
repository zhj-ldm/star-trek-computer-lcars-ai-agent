# -*- coding: utf-8 -*-
"""
web_ui.py - 语音助手 Web 管理界面（Flask）

- 真正的运行主体是 voice_assistant.py（语音链路），本界面只是管理/聊天入口，
  关闭本界面不影响语音助手功能。
- 复用 voice_assistant.py 的 LLMHandler（多会话 + 分层记忆 + 对话持久化），
  共享 config.json 与 对话/ 目录。
- 提供：设置界面（模型/录音/记忆配置）、聊天界面（多对话 + 实时 AI 回复与工具日志）。

用法:
  ~/agentscope-env/bin/python web_ui.py
  浏览器打开 http://127.0.0.1:5000
"""
import os
import sys
import json
import time
import threading
import logging
from collections import deque

# 复用 voice_assistant.py 的 LLMHandler / build_agent / load_config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import voice_assistant as va

from flask import Flask, request, jsonify, render_template, Response

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.json")

app = Flask(__name__)

# ---------- 全局事件队列（SSE 广播） ----------
_events = deque()
_events_lock = threading.Lock()
_events_seq = 0
_current_chat = {"name": None}   # 当前正在处理的对话（日志归属）


def publish_event(etype, data):
    """向所有 SSE 连接广播事件"""
    global _events_seq
    with _events_lock:
        _events.append((_events_seq, etype, data))
        _events_seq += 1
        while len(_events) > 2000:
            _events.popleft()


def event_stream():
    """SSE 生成器：每个连接从自己的游标开始读，避免重复消费"""
    last_id = -1
    while True:
        with _events_lock:
            new_events = [(i, t, d) for i, t, d in _events if i > last_id]
            if new_events:
                last_id = new_events[-1][0]
        for _, etype, data in new_events:
            yield f"event: {etype}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
        time.sleep(0.2)


# ---------- 配置读写 ----------
def load_config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"config.json 解析失败: {e}")
    return {}


def save_config(updates):
    cfg = load_config()
    cfg.update(updates)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


# ---------- LLMHandler 懒加载单例 ----------
_handler = None
_handler_lock = threading.Lock()


# ---------- 删除操作的 Web UI 确认机制 ----------
# 当 agent 遇到删除类 Bash 命令时，VoicePermissionMiddleware 会调用
# confirm_callback；这里用一个共享的 threading.Event + 结果槽把
# 等待流程跨线程桥接到聊天页面：SSE 推送确认提示 + 前端渲染
# CONFIRM / CANCEL 按钮 + 用户点击后 POST /api/confirm/<ok> 回调。
_pending_confirm_lock = threading.Lock()
_pending_confirm = None   # {"id": str, "desc": str, "chat": str, "result": None|bool, "event": Event}
_pending_confirm_seq = 0


def _publish_confirm_prompt(desc):
    """在线程里阻塞式等待用户确认，返回 True/False。若 120s 内无操作则默认拒绝。"""
    global _pending_confirm_seq
    with _pending_confirm_lock:
        _pending_confirm_seq += 1
        cid = f"confirm_{_pending_confirm_seq}_{int(time.time())}"
        ev = threading.Event()
        chat_name = _current_chat.get("name")
        info = {"id": cid, "desc": desc, "chat": chat_name, "result": None, "event": ev}
        _pending_confirm = info
    publish_event("confirm_prompt", {
        "id": cid,
        "chat": chat_name,
        "desc": desc,
        "message": f"This tool wants to run. Confirm or cancel?",
    })
    ok = ev.wait(timeout=120)
    if not ok:
        # 超时清理 + 默认拒绝
        with _pending_confirm_lock:
            if _pending_confirm and _pending_confirm.get("id") == cid:
                _pending_confirm = None
        publish_event("confirm_closed", {"id": cid, "chat": chat_name, "timed_out": True})
        return False
    # event 被 set 了；读结果并清理
    with _pending_confirm_lock:
        current = _pending_confirm
        result = False
        if current and current.get("id") == cid:
            result = bool(current.get("result"))
            _pending_confirm = None
    publish_event("confirm_closed", {"id": cid, "chat": chat_name, "result": result})
    return result


def resolve_confirm(cid, ok: bool):
    """由 HTTP 接口调用，触发等待线程继续执行并给出结果。"""
    with _pending_confirm_lock:
        if not _pending_confirm or _pending_confirm.get("id") != cid:
            return False
        _pending_confirm["result"] = bool(ok)
        _pending_confirm["event"].set()
        return True


def get_handler():
    """懒加载全局 LLMHandler（不启动语音链路，仅复用会话与 agent 能力）"""
    global _handler
    with _handler_lock:
        if _handler is None:
            cfg = load_config()
            api_key = cfg.get("api_key") or os.environ.get("OPENAI_API_KEY", "").strip()
            base_url = cfg.get("base_url")
            model = cfg.get("model")
            if not api_key or not model:
                raise RuntimeError("config.json 缺少 api_key 或 model，请先在设置界面配置")

            def agent_factory(confirm_callback=None, system_prompt=None):
                # Web 界面始终使用「弹 UI 确认」回调，即使外部显式传了
                # confirm_callback=None 也会走默认的 UI 确认路径。
                cb = confirm_callback if confirm_callback is not None else _publish_confirm_prompt
                return va.build_agent(api_key, base_url, model,
                                      confirm_callback=cb,
                                      system_prompt=system_prompt)

            class _Args:
                pass

            _handler = va.LLMHandler(agent_factory, _Args())
            # 注入工具日志回调 → SSE
            va._set_log_callback(lambda text: publish_event(
                "log", {"chat": _current_chat["name"], "text": text}))
            logging.info("LLMHandler 已初始化")
    return _handler


def process_message(chat, message):
    """后台线程：调用 LLMHandler 流式处理，事件推送到 SSE"""
    _current_chat["name"] = chat
    publish_event("user", {"chat": chat, "text": message})
    try:
        handler = get_handler()
        for token in handler.chat_stream(message):
            if token is None:
                break
            publish_event("assistant", {"chat": chat, "text": token})
        publish_event("done", {"chat": chat})
    except Exception as e:
        logging.error(f"处理消息失败: {e}")
        publish_event("error", {"chat": chat, "text": f"处理失败: {e}"})
        publish_event("done", {"chat": chat})
    finally:
        _current_chat["name"] = None


# ---------- 页面路由 ----------
@app.route("/")
def index():
    return render_template("setup.html")


@app.route("/setup")
def setup_page():
    return render_template("setup.html")


@app.route("/chat")
def chat_page():
    return render_template("chat.html")


# ---------- 配置接口 ----------
@app.route("/api/config", methods=["GET"])
def api_config_get():
    cfg = load_config()
    # config.json 是扁平结构：model / api_key / base_url 在顶层，
    # 另外也兼容嵌套结构 cfg["model"] 以处理旧格式。
    model_block = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    return jsonify({
        "model": {
            "model": cfg.get("model") if isinstance(cfg.get("model"), str) else model_block.get("model", ""),
            "api_key": cfg.get("api_key") if isinstance(cfg.get("api_key"), str) else model_block.get("api_key", ""),
            "base_url": cfg.get("base_url") if isinstance(cfg.get("base_url"), str) else model_block.get("base_url", ""),
        },
        "audio": cfg.get("audio", {}),
        "memory": cfg.get("memory", {}),
    })


@app.route("/api/config/model", methods=["POST"])
def api_config_model():
    body = request.get_json() or {}
    save_config({
        "model": (body.get("model") or "").strip(),
        "api_key": (body.get("api_key") or "").strip(),
        "base_url": (body.get("base_url") or "").strip(),
    })
    # 重置 LLMHandler 单例，下次调用会用新配置重建
    global _handler
    with _handler_lock:
        _handler = None
    return jsonify(ok=True)


@app.route("/api/config/audio", methods=["POST"])
def api_config_audio():
    body = request.get_json() or {}
    try:
        audio = {
            "silence_threshold": float(body.get("silence_threshold")),
            "silence_sec": float(body.get("silence_sec")),
            "max_record_sec": float(body.get("max_record_sec")),
            "min_record_sec": float(body.get("min_record_sec")),
        }
    except (TypeError, ValueError):
        return jsonify(ok=False, error="录音参数格式错误")
    save_config({"audio": audio})
    return jsonify(ok=True)


@app.route("/api/config/memory", methods=["POST"])
def api_config_memory():
    body = request.get_json() or {}
    try:
        memory = {
            "recent_keep": int(body.get("recent_keep")),
            "summary_trigger": int(body.get("summary_trigger")),
            "retrieve_top": int(body.get("retrieve_top")),
        }
    except (TypeError, ValueError):
        return jsonify(ok=False, error="记忆参数格式错误")
    save_config({"memory": memory})
    return jsonify(ok=True)


# ---------- 对话接口 ----------
@app.route("/api/chats", methods=["GET"])
def api_chats():
    handler = get_handler()
    return jsonify({"chats": handler.get_chats_sorted()})


@app.route("/api/chats/new", methods=["POST"])
def api_chats_new():
    handler = get_handler()
    handler.new_chat()
    return jsonify({"ok": True, "name": handler._current})


@app.route("/api/chats/<name>/history", methods=["GET"])
def api_chats_history(name):
    handler = get_handler()
    history = handler.get_history(name)
    if history is None:
        return jsonify({"ok": False, "error": "对话不存在"})
    return jsonify({"ok": True, "history": history})


@app.route("/api/chats/<name>/send", methods=["POST"])
def api_chats_send(name):
    body = request.get_json() or {}
    message = (body.get("message") or "").strip()
    if not message:
        return jsonify({"ok": False, "error": "消息为空"})
    handler = get_handler()
    if handler.get_history(name) is None:
        return jsonify({"ok": False, "error": "对话不存在"})
    handler.switch_chat(name)
    threading.Thread(target=process_message, args=(name, message), daemon=True).start()
    return jsonify({"ok": True})


# ---------- 删除操作的 Web 确认接口 ----------
@app.route("/api/confirm/<ok>", methods=["POST"])
def api_confirm(ok):
    """前端点击 CONFIRM / CANCEL 按钮时 POST 这里。
    <ok> 路径段：'1' / 'confirm' / 'yes' → 确认；其他（'0'/'cancel'/'no'）→ 取消。
    body JSON 需要带 id 字段，对应 confirm_prompt 事件里的 id。"""
    body = request.get_json() or {}
    cid = body.get("id")
    approved = str(ok).lower() in ("1", "confirm", "yes", "true", "ok")
    if not cid:
        return jsonify({"ok": False, "error": "missing id"}), 400
    matched = resolve_confirm(cid, approved)
    return jsonify({"ok": True, "matched": matched, "approved": approved})


# ---------- SSE 事件流 ----------
@app.route("/api/stream")
def api_stream():
    return Response(event_stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print("=" * 50)
    print("语音助手 Web 管理界面")
    print("  设置界面: http://127.0.0.1:7689/setup")
    print("  聊天界面: http://127.0.0.1:7689/chat")
    print("  关闭本窗口不影响语音助手功能")
    print("=" * 50)
    app.run(host="127.0.0.1", port=7689, debug=False, threaded=True)
