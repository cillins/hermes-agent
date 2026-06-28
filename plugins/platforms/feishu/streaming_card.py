from __future__ import annotations

import asyncio
import json
import queue
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from gateway.platforms.base import SendResult


THINK_TAG_RE = re.compile(r"</?(?:think|thinking)>", re.IGNORECASE)
FENCE_RE = re.compile(r"^\s*```")
TABLE_SEPARATOR_RE = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$",
    re.MULTILINE,
)
TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
MAX_CARD_TABLES = 5
MAIN_CONTENT_CHUNK_CHARS = 2400
UPDATE_MIN_INTERVAL_SECONDS = 0.5
SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")


@dataclass
class ToolState:
    tool_id: str
    name: str
    status: str
    detail: str = ""


@dataclass
class CardSession:
    conversation_id: str
    message_id: str
    chat_id: str
    status: str = "thinking"
    thinking_text: str = ""
    answer_text: str = ""
    tools: dict[str, ToolState] = field(default_factory=dict)
    tool_call_count: int = 0
    tokens: dict[str, Any] = field(default_factory=dict)
    model: str = "Unknown"
    context: dict[str, Any] = field(default_factory=dict)
    duration: float = 0.0

    @property
    def visible_main_text(self) -> str:
        if self.status in {"completed", "failed"}:
            return self.answer_text
        return self.answer_text or self.thinking_text


class StreamingTextNormalizer:
    def __init__(self) -> None:
        self._pending = ""

    def feed(self, delta: str) -> str:
        text = self._pending + (delta or "")
        safe, self._pending = self._split_safe_text(text)
        return normalize_stream_text(safe)

    @staticmethod
    def _split_safe_text(text: str) -> tuple[str, str]:
        lower = text.lower()
        pending_len = 0
        for tag in ("<think>", "</think>", "<thinking>", "</thinking>"):
            for prefix_len in range(1, len(tag)):
                if lower.endswith(tag[:prefix_len]):
                    pending_len = max(pending_len, prefix_len)
        if pending_len:
            return text[:-pending_len], text[-pending_len:]
        return text, ""


class FeishuStreamingCardConsumer:
    """Native Feishu/Lark interactive-card consumer for one Hermes turn."""

    _DONE = object()
    is_feishu_streaming_card = True

    def __init__(
        self,
        adapter: Any,
        chat_id: str,
        *,
        metadata: Optional[dict[str, Any]] = None,
        initial_reply_to_id: Optional[str] = None,
        title: str = "Hermes Agent",
        footer_fields: Optional[list[str]] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        session_id: str = "",
    ) -> None:
        self.adapter = adapter
        self.chat_id = chat_id
        self.metadata = dict(metadata or {})
        if initial_reply_to_id and "reply_to_message_id" not in self.metadata:
            self.metadata["reply_to_message_id"] = initial_reply_to_id
        self.title = title or "Hermes Agent"
        self.footer_fields = footer_fields
        self.loop = loop or asyncio.get_running_loop()
        self.session = CardSession(
            conversation_id=session_id or chat_id,
            message_id=initial_reply_to_id or session_id or chat_id,
            chat_id=chat_id,
        )
        self._queue: queue.Queue[Any] = queue.Queue()
        self._answer_normalizer = StreamingTextNormalizer()
        self._thinking_normalizer = StreamingTextNormalizer()
        self._message_id: Optional[str] = None
        self._last_update_at = 0.0
        self._started = False
        self._final_response_sent = False
        self._final_content_delivered = False
        self._created_at = time.monotonic()

    @property
    def message_id(self) -> str | None:
        return self._message_id

    @property
    def already_sent(self) -> bool:
        return self._message_id is not None

    @property
    def final_response_sent(self) -> bool:
        return self._final_response_sent

    @property
    def final_content_delivered(self) -> bool:
        return self._final_content_delivered

    def on_delta(self, text: str) -> None:
        if text:
            self._put_threadsafe(("answer", text))

    def on_commentary(self, text: str) -> None:
        if text:
            self._put_threadsafe(("thinking_block", text))

    def on_segment_break(self) -> None:
        return

    def on_tool_event(
        self,
        event_type: str,
        tool_name: str | None = None,
        preview: str | None = None,
        args: Optional[dict[str, Any]] = None,
        **_: Any,
    ) -> None:
        if event_type not in {"tool.started", "tool.completed"}:
            return
        status = "completed" if event_type == "tool.completed" else "running"
        name = tool_name or "tool"
        detail = preview or _preview_args(args)
        self._put_threadsafe(("tool", name, status, detail))

    def finish(
        self,
        final_response: str | None = None,
        *,
        failed: bool = False,
        duration: float = 0.0,
        model: str | None = None,
        tokens: Optional[dict[str, Any]] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> None:
        self._put_threadsafe(
            (
                "finish",
                final_response or "",
                failed,
                duration,
                model or "",
                dict(tokens or {}),
                dict(context or {}),
            )
        )

    async def run(self) -> None:
        await self._ensure_started()
        while True:
            try:
                event = self._queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.08)
                continue
            if event is self._DONE:
                return
            await self._apply_event(event)

    def _put_threadsafe(self, item: Any) -> None:
        try:
            self._queue.put_nowait(item)
        except Exception:
            return

    async def _ensure_started(self) -> bool:
        if self._started:
            return self._message_id is not None
        self._started = True
        result = await self.adapter.send_streaming_card(
            self.chat_id,
            _render_card(self.session, title=self.title, footer_fields=self.footer_fields),
            metadata=self.metadata,
        )
        if getattr(result, "success", False):
            self._message_id = getattr(result, "message_id", None)
            self._last_update_at = time.monotonic()
        return self._message_id is not None

    async def _apply_event(self, event: Any) -> None:
        kind = event[0]
        if kind == "answer":
            self.session.answer_text += self._answer_normalizer.feed(event[1])
            await self._update_card()
        elif kind == "thinking_block":
            text = normalize_stream_text(event[1]).strip()
            if text:
                if self.session.thinking_text:
                    self.session.thinking_text = self.session.thinking_text.rstrip() + "\n\n" + text
                else:
                    self.session.thinking_text = text
                await self._update_card()
        elif kind == "tool":
            _, name, status, detail = event
            self.session.tool_call_count += 1
            self.session.tools[name] = ToolState(name, name, status, detail)
            await self._update_card()
        elif kind == "finish":
            _, final_response, failed, duration, model, tokens, context = event
            self.session.status = "failed" if failed else "completed"
            normalized = normalize_stream_text(final_response).strip()
            if normalized:
                self.session.answer_text = normalized
            elif failed and not self.session.answer_text:
                self.session.answer_text = "消息处理失败"
            self.session.duration = duration
            if model:
                self.session.model = model
            self.session.tokens = tokens
            self.session.context = context
            ok = await self._update_card(force=True)
            self._final_response_sent = ok and not failed and bool(self.session.answer_text.strip())
            self._final_content_delivered = self._final_response_sent
            self._queue.put_nowait(self._DONE)

    async def _update_card(self, *, force: bool = False) -> bool:
        if not await self._ensure_started():
            return False
        now = time.monotonic()
        if not force and now - self._last_update_at < UPDATE_MIN_INTERVAL_SECONDS:
            return True
        card = _render_card(self.session, title=self.title, footer_fields=self.footer_fields)
        result = await self.adapter.update_streaming_card(
            self.chat_id,
            self._message_id or "",
            card,
            finalize=self.session.status in {"completed", "failed"},
        )
        if getattr(result, "success", False):
            self._last_update_at = time.monotonic()
            return True
        return False


def normalize_stream_text(text: str) -> str:
    return THINK_TAG_RE.sub("", text or "")


def _preview_args(args: Optional[dict[str, Any]]) -> str:
    if not args:
        return ""
    try:
        return json.dumps(args, ensure_ascii=False, default=str)[:300]
    except Exception:
        return str(args)[:300]


def _render_card(
    session: CardSession,
    *,
    title: str = "Hermes Agent",
    footer_fields: Optional[list[str]] = None,
) -> dict[str, Any]:
    status = _render_status(session)
    main_text = normalize_stream_text(session.visible_main_text) or (
        "正在思考..." if session.status == "thinking" else ""
    )
    elements = _render_main_content_elements(main_text)
    elements.extend(
        [
            {"tag": "hr", "element_id": "main_divider"},
            {"tag": "markdown", "element_id": "tool_summary", "content": _render_tool_summary(session)},
            {
                "tag": "markdown",
                "element_id": "footer",
                "content": _render_footer(session, footer_fields),
                "text_size": "x-small",
            },
        ]
    )
    return {
        "schema": "2.0",
        "config": {"update_multi": True, "summary": {"content": status["subtitle"]}},
        "header": {
            "template": status["template"],
            "title": {"tag": "plain_text", "content": title or "Hermes Agent"},
            "subtitle": {"tag": "plain_text", "content": status["subtitle"]},
        },
        "body": {"elements": elements},
    }


def _render_status(session: CardSession) -> dict[str, str]:
    if session.status == "completed":
        return {"subtitle": "已完成", "template": "green"}
    if session.status == "failed":
        return {"subtitle": "处理失败", "template": "red"}
    return {"subtitle": "思考中", "template": "indigo"}


def _render_main_content_elements(main_text: str) -> list[dict[str, Any]]:
    table_count = len(re.findall(r"^\|[-: ]+\|", main_text, re.MULTILINE))
    if table_count > MAX_CARD_TABLES:
        matches = list(re.finditer(r"^\|[-: ]+\|", main_text, re.MULTILINE))
        cutoff = matches[MAX_CARD_TABLES - 1].end()
        main_text = main_text[:cutoff].rstrip() + "\n\n> 内容含超过 5 个表格，超出部分已省略。"
    return [
        {"tag": "markdown", "element_id": "main_content" if i == 0 else f"main_content_{i}", "content": chunk}
        for i, chunk in enumerate(split_markdown_blocks(main_text, MAIN_CONTENT_CHUNK_CHARS))
    ]


def _render_tool_summary(session: CardSession) -> str:
    if not session.tools:
        return "工具调用 0 次"
    lines = [f"工具调用 {session.tool_call_count} 次"]
    for tool in session.tools.values():
        lines.append(f"- `{tool.name}`: {tool.status}")
    return "\n".join(lines)


def _render_footer(session: CardSession, footer_fields: Optional[list[str]]) -> str:
    if session.status == "failed":
        return "已停止"
    if session.status != "completed":
        frame = SPINNER_FRAMES[int(time.time() * 8) % len(SPINNER_FRAMES)]
        return f"{frame} 生成中"
    fields = footer_fields or ["duration", "model", "input_tokens", "output_tokens", "context"]
    tokens = session.tokens or {}
    context = session.context or {}
    used_context = _safe_int(context.get("used_tokens"))
    max_context = _safe_int(context.get("max_tokens"))
    context_percent = round(used_context / max_context * 100) if max_context > 0 else 0
    values = {
        "duration": _format_duration(session.duration),
        "model": session.model or "Unknown",
        "input_tokens": f"↑{_format_count(_safe_int(tokens.get('input_tokens')))}",
        "output_tokens": f"↓{_format_count(_safe_int(tokens.get('output_tokens')))}",
        "context": f"ctx {_format_count(used_context)}/{_format_count(max_context)} {context_percent}%",
    }
    selected = [values[key] for key in fields if values.get(key)]
    return " · ".join(selected) if selected else values["duration"]


def split_markdown_blocks(text: str, max_block_size: int) -> list[str]:
    if not text:
        return [""]
    if max_block_size <= 0 or len(text) <= max_block_size:
        return [text]
    blocks = _markdown_structure_blocks(text)
    chunks: list[str] = []
    current = ""
    for block in blocks:
        if len(block) > max_block_size and _is_fenced_code_block(block):
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_fenced_code_block(block, max_block_size))
            continue
        if len(block) > max_block_size and _is_table_block(block):
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_table_block(block, max_block_size))
            continue
        if len(block) > max_block_size and not _is_structured_markdown_block(block):
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_plain_block(block, max_block_size))
            continue
        if current and len(current) + len(block) > max_block_size:
            chunks.append(current)
            current = block
        else:
            current += block
    if current:
        chunks.append(current)
    return chunks or [""]


def _markdown_structure_blocks(text: str) -> list[str]:
    lines = text.splitlines(keepends=True)
    blocks: list[str] = []
    paragraph: list[str] = []
    index = 0

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            blocks.append("".join(paragraph))
            paragraph = []

    while index < len(lines):
        line = lines[index]
        if FENCE_RE.match(line):
            flush_paragraph()
            code = [line]
            index += 1
            while index < len(lines):
                code.append(lines[index])
                if FENCE_RE.match(lines[index]):
                    index += 1
                    break
                index += 1
            blocks.append("".join(code))
            continue
        if TABLE_ROW_RE.match(line) and index + 1 < len(lines) and TABLE_SEPARATOR_RE.match(lines[index + 1]):
            flush_paragraph()
            table = [line, lines[index + 1]]
            index += 2
            while index < len(lines) and TABLE_ROW_RE.match(lines[index]):
                table.append(lines[index])
                index += 1
            blocks.append("".join(table))
            continue
        paragraph.append(line)
        index += 1
        if line.strip() == "":
            flush_paragraph()
    flush_paragraph()
    return blocks or [text]


def _is_structured_markdown_block(block: str) -> bool:
    return "```" in block or TABLE_SEPARATOR_RE.search(block) is not None


def _is_fenced_code_block(block: str) -> bool:
    lines = block.splitlines(keepends=True)
    return bool(lines) and FENCE_RE.match(lines[0]) is not None


def _is_table_block(block: str) -> bool:
    lines = block.splitlines(keepends=True)
    return len(lines) >= 2 and TABLE_ROW_RE.match(lines[0]) is not None and TABLE_SEPARATOR_RE.match(lines[1]) is not None


def _split_fenced_code_block(block: str, max_block_size: int) -> list[str]:
    lines = block.splitlines(keepends=True)
    if len(lines) < 2:
        return _split_plain_block(block, max_block_size)
    opening = lines[0]
    closing = lines[-1] if FENCE_RE.match(lines[-1]) else "```\n"
    body_lines = lines[1:-1] if closing == lines[-1] else lines[1:]
    overhead = len(opening) + len(closing)
    if overhead >= max_block_size:
        return _split_plain_block(block, max_block_size)
    body_limit = max_block_size - overhead
    chunks: list[str] = []
    current = ""
    for line in body_lines:
        if current and len(current) + len(line) > body_limit:
            chunks.append(_wrap_code_chunk(opening, current, closing))
            current = ""
        if len(line) > body_limit:
            chunks.extend(_wrap_code_chunk(opening, piece, closing) for piece in _split_plain_block(line, body_limit))
            continue
        current += line
    if current or not chunks:
        chunks.append(_wrap_code_chunk(opening, current, closing))
    return chunks


def _wrap_code_chunk(opening: str, body: str, closing: str) -> str:
    if body and not body.endswith("\n"):
        body += "\n"
    return opening + body + closing


def _split_table_block(block: str, max_block_size: int) -> list[str]:
    lines = block.splitlines(keepends=True)
    if len(lines) < 3:
        return _split_plain_block(block, max_block_size)
    header = "".join(lines[:2])
    rows = lines[2:]
    if len(header) >= max_block_size:
        return _split_plain_block(block, max_block_size)
    row_limit = max_block_size - len(header)
    chunks: list[str] = []
    current = ""
    for row in rows:
        if current and len(current) + len(row) > row_limit:
            chunks.append(header + current)
            current = ""
        if len(row) > row_limit:
            if current:
                chunks.append(header + current)
                current = ""
            chunks.extend(header + piece for piece in _split_plain_block(row, row_limit))
            continue
        current += row
    if current or not chunks:
        chunks.append(header + current)
    return chunks


def _split_plain_block(block: str, max_block_size: int) -> list[str]:
    chunks: list[str] = []
    remaining = block
    while len(remaining) > max_block_size:
        split_at = max(remaining.rfind(" ", 0, max_block_size + 1), remaining.rfind("\n", 0, max_block_size + 1))
        if split_at <= 0:
            split_at = max_block_size
        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:]
    if remaining:
        chunks.append(remaining)
    return chunks


def _safe_int(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return max(number, 0)


def _format_duration(seconds: float) -> str:
    try:
        total = max(0, int(round(float(seconds))))
    except (TypeError, ValueError):
        total = 0
    minutes, sec = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes}m{sec}s"
    if minutes:
        return f"{minutes}m{sec}s"
    return f"{sec}s"


def _format_count(value: int) -> str:
    if value >= 1_000_000:
        return _format_scaled(value, 1_000_000, "m")
    if value >= 1_000:
        return _format_scaled(value, 1_000, "k")
    return str(value)


def _format_scaled(value: int, factor: int, suffix: str) -> str:
    scaled = value / factor
    if scaled >= 100 or scaled.is_integer():
        return f"{int(round(scaled))}{suffix}"
    return f"{scaled:.1f}".rstrip("0").rstrip(".") + suffix


async def send_streaming_card(adapter: Any, chat_id: str, card: dict[str, Any], metadata: Optional[dict[str, Any]]) -> SendResult:
    response = await adapter._feishu_send_with_retry(
        chat_id=chat_id,
        msg_type="interactive",
        payload=json.dumps(card, ensure_ascii=False),
        reply_to=(metadata or {}).get("reply_to_message_id"),
        metadata=metadata,
    )
    return adapter._finalize_send_result(response, "streaming card send failed")


async def update_streaming_card(adapter: Any, message_id: str, card: dict[str, Any]) -> SendResult:
    body = adapter._build_update_message_body(
        msg_type="interactive",
        content=json.dumps(card, ensure_ascii=False),
    )
    request = adapter._build_update_message_request(message_id=message_id, request_body=body)
    response = await adapter._run_blocking(adapter._client.im.v1.message.update, request)
    result = adapter._finalize_send_result(response, "streaming card update failed")
    if result.success:
        result.message_id = message_id
    return result
