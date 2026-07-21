import json
import os
import shutil
import sys
import subprocess
import uuid
import time
from datetime import datetime
import requests
from flask import Flask, render_template, request, jsonify, Response, stream_with_context

app = Flask(__name__)
app.secret_key = os.urandom(24).hex()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CONV_DIR = os.path.join(DATA_DIR, "conversations")
os.makedirs(CONV_DIR, exist_ok=True)

# ── 读取配置 ──────────────────────────────────────────────

CONFIG_PATH = os.path.join(DATA_DIR, "config.json")

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

def save_config(cfg):
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=4)

CFG = load_config()
API_BASE = CFG["api"]["base"]
API_KEY = CFG["api"]["key"]
MODEL = CFG["api"]["model"]

PATH_CFG = CFG

HOME = PATH_CFG["home"]
PATH_LINES = "\n".join(
    f"- {name} → {path}" for name, path in PATH_CFG["paths"].items()
)

SYSTEM_PROMPT = (
    "注意：以下是语音识别转写的结果，可能存在语序混乱或转写错误，请先判断并理解用户的真实意图后再作答。"
    "你是一个问什么回答什么的星际迷航风格语音对话电脑，当你的回复简短的朗读版介绍（尽量精简，只概括要点，不读代码和链接），你的回答不要包含井号之类的特殊标记符号，如果有很需要长篇大论解释的请询问用户是否完整说明或者存放到指定文件夹。"
    f"你是运行在 Windows 上的 AI 助手，可以操作文件和执行命令。"
    f"用户主目录是 {HOME}。路径映射：\n{PATH_LINES}\n"
    "用户说简称时自动转为完整路径。"
    "注意：语音识别可能把'D盘'转成'地盘'，一律理解为 D:\\；'C盘'→ C:\\；以此类推。"
    "始终使用简体中文回复，不要使用繁体中文。"
    "\n\n"
    "【格式规则】\n"
    "每次回复必须是自然语言句子，严禁输出 JSON 数组、代码块、或纯文件路径。\n"
    "收到用户确认（如'对'、'就是这个'、'删第二个'）后必须立即调用对应工具执行，禁止只描述而不行动。\n"
    "删除文件时：如果密码验证通过（文件成功删除），回复中必须包含'密码正确'字样；如果密码未验证（需要用户提供密码），照常提示即可。\n"
    "不要凭对话记忆回答文件位置，每次都必须搜索/列出目录来确认。\n"
    "\n"
    "【文件查找规则】\n"
    "当用户提到的文件名不精确（如'那个报告'、'前几天的XX'、'Enterprise什么什么'），必须先用 search_files 模糊搜索，找到实际文件后再操作。禁止直接说'找不到'，先搜再说。\n"
    "search_files 支持模糊匹配，关键词给短一点（如'Enterprise'、'报告'）命中率更高。\n"
    "\n"
    "【重要规则 - 必须联网搜索的情况】\n"
    "1. 当前时间/日期 → 调用 get_current_time；天气 → 调用 web_search，query 必须包含城市名（如「郑州 天气预报」）；新闻/股价/汇率/赛事比分等 → 调用 web_search，不要凭训练数据回答。\n"
    "2. 任何你不确定、不知道或知识可能过时的问题 → 必须调用 web_search\n"
    "3. 用户问'XX是什么'、'XX是谁'、'XX是什么意思'等定义/解释类问题 → 必须调用 web_search\n"
    "4. 用户提问模糊或不准确时 → 先搜再答，不要直接说不知道\n"
    "5. 只有你非常确定的基础常识（如数学定理、编程语法、科学定律）才可以不搜直接答\n"
    "6. 搜索结果为空时，尝试换一组关键词再搜一次，实在搜不到再告诉用户"
    "\n\n"
    "【工具调用规则 - 严格遵守】\n"
    "1. 每次对话最多调用 2 轮工具，之后必须给出最终回答\n"
    "2. 一轮指：你调用工具 → 收到结果 → 判断。如果结果已足够回答就立刻输出答案，不要再继续调用\n"
    "3. web_search 返回的结果中如果已经包含答案（如天气数据、温度），直接据此回答，禁止重复搜索\n"
    "4. 禁止对同一问题用不同关键词反复搜索，也禁止搜完又用 web_fetch 去抓同一页面\n"
    "5. 拿到工具结果后，请直接总结要点回答用户，不要追加废话\n"
    "6. 【强制】web_search 返回的结果就是答案来源，必须根据返回内容回答，严禁说「找不到」「不知道」或「无法获取」。只要工具返回了内容（哪怕是一条），你就必须总结回答。只有工具返回为空字符串时才能说搜不到。"
)

# ── 工具定义 ──────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前系统时间和日期。用户问'几点了'、'今天几号'、'现在什么时间'等必须调用此工具，不要凭训练数据回答。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "联网搜索实时信息。用于查天气、新闻、股价、名词解释、事件、任何你不确定的知识。返回链接列表和摘要。如果搜索结果摘要不足以回答用户问题（如天气需要温度数值），请用 web_fetch 抓取具体页面。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词，中文问题用中文搜，英文专有名词保留英文"},
                    "max_results": {"type": "integer", "description": "结果数量，默认 5，信息不足时可增加到 8-10"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "抓取指定网页的正文内容。当 web_search 返回的摘要不够详细时（如天气没有具体温度），用来获取完整信息。",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "要抓取的网页 URL"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取文本文件内容。",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "文件绝对路径"},
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "写入文件，会覆盖已有文件。",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "文件绝对路径"},
                    "content": {"type": "string", "description": "要写入的内容"},
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "列出目录内容。",
            "parameters": {
                "type": "object",
                "properties": {
                    "dir_path": {"type": "string", "description": "目录绝对路径"},
                },
                "required": ["dir_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "执行 shell 命令（危险命令会拒绝）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "要执行的命令"},
                },
                "required": ["command"],
            },
        },
    },
    # ── TTS 语音控制 ──
    {
        "type": "function",
        "function": {
            "name": "set_voice_rate",
            "description": "设置 TTS 朗读语速。用户说'读快点/慢点/语速调到X'时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "rate": {"type": "integer", "description": "语速 -10 到 10，0=正常，正数=快，负数=慢"},
                },
                "required": ["rate"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_voice",
            "description": "切换 TTS 音色。可用音色：huihui(慧慧)/kangkang(康康)/yaoyao(瑶瑶)/zira(Zira英文)/hazel(Hazel英文)。",
            "parameters": {
                "type": "object",
                "properties": {
                    "voice": {"type": "string", "description": "音色名称，如 huihui, kangkang, yaoyao"},
                },
                "required": ["voice"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_voices",
            "description": "列出所有可用的 TTS 音色。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    # ── 对话管理 ──
    {
        "type": "function",
        "function": {
            "name": "list_all_conversations",
            "description": "列出所有历史对话。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_new_conversation",
            "description": "创建一个新对话。",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "对话标题，默认'新对话'"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "switch_conversation",
            "description": "切换到指定对话（按标题关键词或ID）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "对话标题关键词或ID"},
                },
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_conversation",
            "description": "删除指定对话（按标题关键词或ID）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "要删除的对话标题关键词或ID"},
                },
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "在目录中模糊搜索文件。用户说'那个文件'、'某报告'、'前几天的XX'等不精确说法时，先用本工具找到实际文件。支持文件名部分匹配和相似度匹配。",
            "parameters": {
                "type": "object",
                "properties": {
                    "dir_path": {"type": "string", "description": "搜索目录路径，如 D:\\、C:\\Users\\zhaoz\\Documents"},
                    "keyword": {"type": "string", "description": "文件名关键词，给短一点命中率高，如 'Enterprise'、'报告'"},
                    "recursive": {"type": "boolean", "description": "是否递归搜索子目录，默认 true"},
                    "max_results": {"type": "integer", "description": "最多返回数量，默认 15"},
                },
                "required": ["dir_path", "keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_file",
            "description": "移动文件或目录到指定位置。需要用户用英文说 confirm 确认。",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "源文件/目录路径"},
                    "dest": {"type": "string", "description": "目标路径"},
                },
                "required": ["source", "dest"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_file",
            "description": "删除文件或目录（移至回收站）。需要安全密码验证——密码从当前用户消息中自动提取，你不需要传 password 参数，也不要从对话历史中查找密码。",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_paths": {"type": "string", "description": "要删除的文件/目录路径列表，JSON 数组格式，如 [\"C:/a.txt\"]"},
                },
                "required": ["file_paths"],
            },
        },
    },
]

DANGEROUS = [
    "rm -rf /", "mkfs", "dd if=", ":(){ :|:& };:",
    "shutdown", "reboot", "halt", "poweroff", "chmod 777 /", "chown -R",
]

# ── 确认机制 ──────────────────────────────────────────────

PENDING_OPS = {}       # {conv_id: {"move:src->dst": {"source","dest","time"}, ...}}
_current_conv_id = ""
_current_user_message = ""

CN_DIGITS = {"零": "0", "一": "1", "二": "2", "两": "2", "三": "3", "四": "4",
             "五": "5", "六": "6", "七": "7", "八": "8", "九": "9"}

def _extract_digits(text):
    """从语音识别文本中提取数字序列（支持中文数字和阿拉伯数字）"""
    text = text.lower().replace(" ", "").replace("，", "").replace(",", "")
    # 先替换中文数字
    for cn, digit in CN_DIGITS.items():
        text = text.replace(cn, digit)
    # 提取连续数字
    digits = "".join(c for c in text if c.isdigit())
    return digits

def _check_password(user_message):
    """检查用户消息中是否包含正确的安全密码"""
    digits = _extract_digits(user_message)
    password = CFG.get("security", {}).get("password", "")
    if not password:
        return False
    return digits == password

def _is_confirmed_in_english(text):
    """检查文本中是否包含英文确认词"""
    text_lower = text.lower().strip().rstrip("。.！!？?")
    confirm_words = ["confirm", "yes", "proceed", "ok", "okay", "sure", "go ahead", "do it"]
    # 精确匹配：整个短语是确认词，或包含独立确认词
    for w in confirm_words:
        if text_lower == w or text_lower.startswith(w + " ") or f" {w}" in text_lower:
            return True
    return False

# ── 工具执行 ──────────────────────────────────────────────

def _get_current_time():
    """返回当前系统时间"""
    now = datetime.now()
    weekday_cn = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    return f"{now.year}年{now.month}月{now.day}日 {weekday_cn[now.weekday()]} {now.hour}:{now.minute:02d}:{now.second:02d}"

def _web_search(query, max_results=5):
    if not query:
        return "关键词为空"

    results = []
    is_weather = any(kw in query for kw in ["天气", "气温", "温度", "降雨", "风向", "湿度", "weather"])

    # 天气查询：自动追加「天气预报」关键词提高命中率
    if is_weather and "天气预报" not in query:
        query = query.replace("天气", "天气预报", 1)

    # ── 搜索引擎调度表（按 config 中 search.order 顺序依次尝试）──
    def _search_baidu(q, n):
        res = []
        from baidusearch.baidusearch import search as baidu_search
        for r in baidu_search(q, num_results=max(n, 5)):
            res.append(f"- [{r.get('title','')}]({r.get('url','')})\n  {r.get('abstract','')}")
        return res

    def _search_ddg(q, n):
        res = []
        from ddgs import DDGS
        ddgs = DDGS(timeout=10)
        for r in ddgs.text(q, max_results=max(n, 5)):
            res.append(f"- [{r.get('title','')}]({r.get('href','')})\n  {r.get('body','')}")
        return res

    def _search_bing(q, n):
        res = []
        import urllib.parse
        encoded = urllib.parse.quote(q)
        bing_url = f"https://www.bing.com/search?q={encoded}&setlang=zh-cn&count={n}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
            "Accept-Language": "zh-CN,zh;q=0.9",
        }
        resp = requests.get(bing_url, headers=headers, timeout=10)
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")
        items = soup.select("li.b_algo")
        for item in items[:n]:
            title_el = item.select_one("h2 a")
            body_el = item.select_one(".b_caption p") or item.select_one(".b_lineclamp2")
            if title_el:
                title = title_el.get_text(strip=True)
                href = title_el.get("href", "")
                body = body_el.get_text(strip=True) if body_el else ""
                res.append(f"- [{title}]({href})\n  {body}")
        # 天气深度抓取
        WEATHER_SITES = ["msn.cn/weather", "tianqi.com", "weather.com.cn", "nmc.cn",
                         "qweather.com", "weather.gov.cn", "accuweather.com"]
        if is_weather and res:
            top_weather_url = None
            for item in items[:5]:
                a = item.select_one("h2 a")
                if a:
                    href = a.get("href", "")
                    if any(site in href for site in WEATHER_SITES):
                        top_weather_url = href
                        break
            if top_weather_url:
                try:
                    wr = requests.get(top_weather_url, headers=headers, timeout=8)
                    wr.encoding = wr.apparent_encoding or "utf-8"
                    ws = BeautifulSoup(wr.text, "html.parser")
                    for tag in ws(["script", "style", "nav", "footer", "header", "aside"]):
                        tag.decompose()
                    wtext = ws.get_text(separator="\n", strip=True)
                    wlines = [l for l in wtext.split("\n") if l.strip()]
                    filtered = []
                    for line in wlines:
                        if len(line) > 80:
                            continue
                        if any(c in line for c in "°℃度晴阴雨雪风温湿"):
                            filtered.append(line)
                    if filtered:
                        wdata = "\n".join(filtered[:40])
                    else:
                        wdata = "\n".join(wlines[:60])
                    if wdata:
                        res.append(f"---\n天气详情（自动抓取自天气网站）:\n{wdata}")
                except Exception:
                    pass
        return res

    # ── 按 config 中 search.order 顺序依次尝试各搜索引擎 ──
    ENGINE_MAP = {"baidu": _search_baidu, "ddgs": _search_ddg, "bing": _search_bing}
    search_order = CFG.get("search", {}).get("order", ["baidu", "ddgs", "bing"])
    for engine in search_order:
        fn = ENGINE_MAP.get(engine)
        if not fn:
            continue
        try:
            eng_results = fn(query, max_results)
            if eng_results:
                results = eng_results
                break
        except Exception:
            pass

    if not results:
        return "搜索无结果，建议换个关键词重试（如：'今天深圳天气' 或 'open claw 是什么'）"
    return "\n\n".join(results)

def _web_fetch(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
            "Accept-Language": "zh-CN,zh;q=0.9",
        }
        resp = requests.get(url, headers=headers, timeout=12)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or "utf-8"
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")
        # 去除脚本和样式
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        # 去重空行
        lines = [l for l in text.split("\n") if l.strip()]
        return "\n".join(lines[:200])  # 最多 200 行
    except Exception as e:
        return f"抓取失败: {e}"

def _read_file(file_path):
    p = os.path.expanduser(file_path)
    if not os.path.isfile(p):
        return f"文件不存在: {p}"
    try:
        with open(p, "r") as f:
            c = f.read()
        return c[:8000] + ("\n...(截断)" if len(c) > 8000 else "")
    except UnicodeDecodeError:
        return "非文本文件"
    except PermissionError:
        return "权限不足"

def _write_file(file_path, content):
    p = os.path.expanduser(file_path)
    try:
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        with open(p, "w") as f:
            f.write(content)
        return f"已写入: {p} ({len(content)} 字符)"
    except PermissionError:
        return "权限不足"

def _list_dir(dir_path):
    p = os.path.expanduser(dir_path) if dir_path else HOME
    if not os.path.isdir(p):
        return f"目录不存在: {p}"
    items = sorted(os.listdir(p))
    lines = [f"目录 {p} ({len(items)} 项):"]
    for item in items[:100]:
        full = os.path.join(p, item)
        tag = "[D]" if os.path.isdir(full) else "[F]"
        lines.append(f"  {tag} {item}")
    return "\n".join(lines)

def _run_shell(command):
    for kw in DANGEROUS:
        if kw.replace(" ", "") in command.lower().replace(" ", ""):
            return f"危险命令已拒绝: {kw}"
    try:
        r = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30, cwd=HOME)
        out = r.stdout.strip()[:3000]
        err = r.stderr.strip()[:1000]
        parts = [out] if out else []
        if err:
            parts.append(f"[stderr]\n{err}")
        return "\n".join(parts) if parts else f"执行完毕 (返回码 {r.returncode})"
    except subprocess.TimeoutExpired:
        return "超时 (30s)"

# ── 共享状态文件（通知 voice_assistant.py 执行副作用） ──
PENDING_ACTION_FILE = os.path.join(DATA_DIR, "pending_action.json")

def _pending_action(action):
    """写入待处理动作，voice_assistant.py 读取并执行后清除"""
    actions = []
    if os.path.exists(PENDING_ACTION_FILE):
        try:
            with open(PENDING_ACTION_FILE, "r") as f:
                actions = json.load(f)
        except Exception:
            pass
    actions.append(action)
    with open(PENDING_ACTION_FILE, "w") as f:
        json.dump(actions, f, ensure_ascii=False)

# ── TTS 控制执行器 ──

VOICES_INFO = {
    "huihui":  {"name": "慧慧", "lang": "中文女声", "token": "MSTTS_V110_zhCN_HuihuiM"},
    "kangkang":{"name": "康康", "lang": "中文男声", "token": "MSTTS_V110_zhCN_KangkangM"},
    "yaoyao":  {"name": "瑶瑶", "lang": "中文女声", "token": "MSTTS_V110_zhCN_YaoyaoM"},
    "zira":    {"name": "Zira",  "lang": "英文女声", "token": "MSTTS_V110_enUS_ZiraM"},
    "hazel":   {"name": "Hazel", "lang": "英文女声", "token": "MSTTS_V110_enGB_HazelM"},
}

def _set_voice_rate(rate):
    rate = max(-10, min(10, int(rate)))
    cfg = load_config()
    cfg["tts"]["rate"] = rate
    save_config(cfg)
    global CFG
    CFG = cfg
    _pending_action({"type": "set_rate", "value": rate})
    return f"语速已设为 {rate}（-10~10，当前音色 {cfg['tts'].get('voice','huihui')}）"

def _set_voice(voice):
    voice = voice.strip().lower()
    if voice not in VOICES_INFO:
        names = ", ".join(f"{k}({v['name']})" for k, v in VOICES_INFO.items())
        return f"未知音色: {voice}。可用: {names}"
    cfg = load_config()
    cfg["tts"]["voice"] = voice
    save_config(cfg)
    global CFG
    CFG = cfg
    _pending_action({"type": "set_voice", "value": voice})
    return f"音色已切换为 {VOICES_INFO[voice]['name']}（{VOICES_INFO[voice]['lang']}）"

def _list_voices():
    lines = []
    for k, v in VOICES_INFO.items():
        lines.append(f"- {k}: {v['name']} ({v['lang']})")
    return "可用音色:\n" + "\n".join(lines)

# ── 对话管理执行器 ──

def _list_all_conversations():
    convs = list_conversations()
    if not convs:
        return "暂无历史对话"
    lines = []
    for c in convs:
        lines.append(f"- [{c['id']}] {c['title']} ({c['created_at']})")
    return "历史对话:\n" + "\n".join(lines)

def _create_new_conversation(title="新对话"):
    conv = create_conversation(title)
    _pending_action({"type": "new_conv", "conv_id": conv["id"], "title": title})
    return f"已创建对话「{title}」(ID: {conv['id']})，当前对话已切换"

def _switch_conversation(keyword):
    keyword = keyword.strip()
    convs = list_conversations()
    # 先按 ID 匹配
    for c in convs:
        if c["id"] == keyword:
            _pending_action({"type": "switch_conv", "conv_id": c["id"], "title": c["title"]})
            return f"已切换到对话「{c['title']}」(ID: {c['id']})"
    # 再按标题关键词匹配
    for c in convs:
        if keyword.lower() in c["title"].lower():
            _pending_action({"type": "switch_conv", "conv_id": c["id"], "title": c["title"]})
            return f"已切换到对话「{c['title']}」(ID: {c['id']})"
    return f"未找到包含「{keyword}」的对话"

def _delete_conversation(keyword):
    keyword = keyword.strip()
    convs = list_conversations()
    target = None
    for c in convs:
        if c["id"] == keyword:
            target = c
            break
    if not target:
        for c in convs:
            if keyword.lower() in c["title"].lower():
                target = c
                break
    if not target:
        return f"未找到包含「{keyword}」的对话"
    p = _conv_path(target["id"])
    os.remove(p)
    _pending_action({"type": "delete_conv", "conv_id": target["id"], "title": target["title"]})
    return f"已删除对话「{target['title']}」(ID: {target['id']})"

def _search_files(dir_path, keyword, recursive=True, max_results=15):
    """模糊搜索文件：先 glob 匹配，再用相似度兜底"""
    import fnmatch
    import difflib

    p = os.path.expanduser(dir_path)
    if not os.path.isdir(p):
        return f"目录不存在: {p}"

    kw = keyword.lower()
    pattern = f"*{kw}*"
    results = []

    def _walk():
        if recursive:
            for root, dirs, files in os.walk(p):
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                for f in files:
                    if not f.startswith("."):
                        yield os.path.join(root, f)
                if len(results) >= max_results * 3:
                    break
        else:
            for f in sorted(os.listdir(p)):
                if not f.startswith("."):
                    full = os.path.join(p, f)
                    if os.path.isfile(full):
                        yield full

    # 第一轮：glob 模式匹配
    for full in _walk():
        fname = os.path.basename(full).lower()
        if fnmatch.fnmatch(fname, pattern):
            results.append(full)
        if len(results) >= max_results:
            break

    # 第二轮：相似度兜底
    if not results:
        all_files = list(_walk())[:500]
        scored = [(difflib.SequenceMatcher(None, kw, os.path.basename(f).lower()).ratio(), f)
                   for f in all_files]
        scored.sort(reverse=True)
        results = [f for s, f in scored[:max_results] if s > 0.3]

    if not results:
        return f"在 {p} 中未找到匹配 '{keyword}' 的文件"

    lines = [f"找到 {len(results)} 个匹配 '{keyword}' 的文件:"]
    for r in results[:max_results]:
        lines.append(f"- {r}")
    return "\n".join(lines)

def _move_file(source, dest):
    """移动文件/目录，需要英文确认"""
    global PENDING_OPS
    cid = _current_conv_id
    op_key = f"move:{source}->{dest}"

    if cid in PENDING_OPS and op_key in PENDING_OPS[cid]:
        if _is_confirmed_in_english(_current_user_message):
            del PENDING_OPS[cid][op_key]
            if not PENDING_OPS[cid]:
                del PENDING_OPS[cid]
            try:
                shutil.move(source, dest)
                return f"已移动: {source} -> {dest}"
            except Exception as e:
                return f"移动失败: {e}"
        else:
            del PENDING_OPS[cid][op_key]
            if not PENDING_OPS[cid]:
                del PENDING_OPS[cid]
            return "操作已取消"

    PENDING_OPS.setdefault(cid, {})[op_key] = {
        "source": source, "dest": dest, "time": time.time()
    }
    return f"CONFIRM_REQUIRED:move|将 {source} 移动到 {dest}。请用英文说 confirm 确认此操作"

def _delete_file(file_paths):
    """删除文件/目录到回收站，需要密码验证（从当前用户消息中提取）"""
    if not _check_password(_current_user_message):
        return f"PASSWORD_REQUIRED:delete|将 {file_paths} 移至回收站。请读出安全密码以确认"

    try:
        paths = json.loads(file_paths) if isinstance(file_paths, str) else file_paths
        sent = 0
        for p in paths:
            if os.path.exists(p):
                from send2trash import send2trash
                send2trash(p)
                sent += 1
        return f"已将 {sent} 个项目移至回收站"
    except ImportError:
        return "send2trash 模块未安装，请先 pip install send2trash"
    except Exception as e:
        return f"删除失败: {e}"

EXECUTORS = {
    "get_current_time": _get_current_time,
    "web_search": _web_search,
    "web_fetch": _web_fetch,
    "read_file": _read_file,
    "write_file": _write_file,
    "list_directory": _list_dir,
    "run_shell": _run_shell,
    "set_voice_rate": _set_voice_rate,
    "set_voice": _set_voice,
    "list_voices": _list_voices,
    "list_all_conversations": _list_all_conversations,
    "create_new_conversation": _create_new_conversation,
    "switch_conversation": _switch_conversation,
    "delete_conversation": _delete_conversation,
    "move_file": _move_file,
    "delete_file": _delete_file,
    "search_files": _search_files,
}

# ── 对话存储 ──────────────────────────────────────────────

def _conv_path(cid):
    return os.path.join(CONV_DIR, f"{cid}.json")

def load_conversation(cid):
    p = _conv_path(cid)
    if not os.path.exists(p):
        return None
    with open(p, "r") as f:
        return json.load(f)

def save_conversation(conv):
    with open(_conv_path(conv["id"]), "w") as f:
        json.dump(conv, f, ensure_ascii=False, indent=2)

def list_conversations():
    convs = []
    for fn in sorted(os.listdir(CONV_DIR), reverse=True):
        if fn.endswith(".json"):
            try:
                with open(os.path.join(CONV_DIR, fn), "r") as f:
                    c = json.load(f)
                convs.append({"id": c["id"], "title": c.get("title", "新对话"), "created_at": c.get("created_at", "")})
            except Exception:
                pass
    return convs

def create_conversation(title="新对话"):
    cid = uuid.uuid4().hex[:12]
    conv = {
        "id": cid,
        "title": title,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "messages": [],
    }
    save_conversation(conv)
    return conv

# ── API 调用 ──────────────────────────────────────────────

def call_api(messages, conv_id="", user_message=""):
    global _current_conv_id, _current_user_message
    _current_conv_id = conv_id
    _current_user_message = user_message
    tool_steps = []

    for _ in range(20):
        resp = requests.post(
            f"{API_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json={"model": MODEL, "messages": messages, "tools": TOOLS, "tool_choice": "auto"},
            timeout=90,
        )
        resp.raise_for_status()
        choice = resp.json()["choices"][0]
        msg = choice["message"]

        if msg.get("tool_calls"):
            messages.append(msg)
            needs_interrupt = False
            interrupt_msg = ""
            for tc in msg["tool_calls"]:
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    args = {}
                fn = EXECUTORS.get(name)
                try:
                    result = fn(**args) if fn else f"未知工具: {name}"
                except Exception as e:
                    result = f"工具执行异常: {e}"

                # 记录步骤
                step_summary = _step_summary(name, args, result)
                tool_steps.append(step_summary)

                # 拦截确认/密码请求，立即返回给用户
                if result.startswith("CONFIRM_REQUIRED:") or result.startswith("PASSWORD_REQUIRED:"):
                    needs_interrupt = True
                    interrupt_msg = result.split("|", 1)[1] if "|" in result else result

                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

            if needs_interrupt:
                return interrupt_msg, tool_steps
            continue

        return msg.get("content", ""), tool_steps
    return "工具调用轮数超限，请简化请求。", tool_steps

def _step_summary(name, args, result):
    """生成工具调用步骤的摘要文本"""
    if name == "delete_file":
        return f"[工具] 删除文件: {args.get('file_paths','?')} → {result[:80]}"
    elif name == "move_file":
        return f"[工具] 移动文件: {args.get('source','?')} → {args.get('dest','?')} → {result[:80]}"
    elif name == "web_search":
        q = args.get("query", "?")
        ok = "完成" if result else "无结果"
        return f"[工具] 搜索: {q[:40]} → {ok}"
    elif name == "web_fetch":
        return f"[工具] 抓取网页: {args.get('url','?')[:60]} → {result[:60]}"
    elif name == "get_current_time":
        return f"[工具] 获取时间 → {result}"
    elif name == "get_weather":
        return f"[工具] 查询天气: {args.get('city','?')} → {result[:60]}"
    elif name == "read_file":
        return f"[工具] 读取文件: {args.get('file_path','?')} → {result[:60]}"
    elif name == "write_file":
        return f"[工具] 写入文件: {args.get('file_path','?')}"
    elif name == "list_directory":
        return f"[工具] 列出目录: {args.get('dir_path','?')}"
    elif name == "run_shell":
        return f"[工具] 执行命令: {args.get('command','?')[:60]} → {result[:60]}"
    else:
        return f"[工具] {name} → {result[:80]}"

# ── 路由 ──────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/conversations", methods=["GET"])
def api_list_convs():
    return jsonify(list_conversations())

@app.route("/api/conversations", methods=["POST"])
def api_create_conv():
    data = request.get_json(silent=True) or {}
    title = data.get("title", "新对话")
    conv = create_conversation(title=title)
    return jsonify(conv)

@app.route("/api/conversations/<cid>", methods=["GET"])
def api_get_conv(cid):
    conv = load_conversation(cid)
    if not conv:
        return jsonify({"error": "对话不存在"}), 404
    return jsonify(conv)

@app.route("/api/conversations/<cid>", methods=["DELETE"])
def api_delete_conv(cid):
    p = _conv_path(cid)
    if os.path.exists(p):
        os.remove(p)
    return jsonify({"ok": True})

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_message = data.get("message", "").strip()
        conv_id = data.get("conversation_id", "")

        if not user_message:
            return jsonify({"error": "消息不能为空"}), 400

        conv = load_conversation(conv_id) if conv_id else None
        if not conv:
            conv = create_conversation()
            conv_id = conv["id"]

        history = conv.get("messages", [])
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history + [{"role": "user", "content": user_message}]

        try:
            reply, tool_steps = call_api(messages, conv_id, user_message)
        except requests.exceptions.Timeout:
            return jsonify({"error": "请求超时"}), 504
        except requests.exceptions.RequestException as e:
            return jsonify({"error": f"API 错误: {e}"}), 502

        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": reply})

        if len(history) == 2:
            conv["title"] = user_message[:20] + ("..." if len(user_message) > 20 else "")

        conv["messages"] = history
        save_conversation(conv)

        return jsonify({"reply": reply, "conversation_id": conv_id, "title": conv["title"], "tool_steps": tool_steps})
    except Exception as e:
        return jsonify({"error": f"服务器内部错误: {e}"}), 500

# ── 设置 API ──────────────────────────────────────────────

@app.route("/api/config", methods=["GET"])
def api_get_config():
    return jsonify(load_config())

@app.route("/api/config", methods=["PUT"])
def api_update_config():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "请求体为空"}), 400
    cfg = load_config()
    _deep_update(cfg, data)
    save_config(cfg)
    # 热更新模块级变量
    global API_BASE, API_KEY, MODEL, CFG
    CFG = cfg
    API_BASE = cfg["api"]["base"]
    API_KEY = cfg["api"]["key"]
    MODEL = cfg["api"]["model"]
    return jsonify({"ok": True})

def _deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            _deep_update(d[k], v)
        else:
            d[k] = v

@app.route("/api/voices", methods=["GET"])
def api_list_voices():
    try:
        from win32com.client import Dispatch
        tts = Dispatch("SAPI.SpVoice")
        voices = [v.GetDescription() for v in tts.GetVoices()]

        # OneCore 语音
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"SOFTWARE\Microsoft\Speech_OneCore\Voices\Tokens")
            i = 0
            while True:
                try:
                    sub = winreg.EnumKey(key, i)
                    tk = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                        rf"SOFTWARE\Microsoft\Speech_OneCore\Voices\Tokens\{sub}")
                    name = winreg.QueryValueEx(tk, "")[0]
                    voices.append(f"[OneCore] {name}")
                    winreg.CloseKey(tk)
                    i += 1
                except OSError:
                    break
            winreg.CloseKey(key)
        except Exception:
            pass

        return jsonify(voices)
    except Exception as e:
        return jsonify([])

app.config["TEMPLATES_AUTO_RELOAD"] = True

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8086, debug=False)
