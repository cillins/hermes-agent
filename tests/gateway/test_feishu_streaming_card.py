from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


from gateway.config import PlatformConfig
from plugins.platforms.feishu.adapter import FeishuAdapter
from plugins.platforms.feishu import streaming_card as streaming_card_module
from plugins.platforms.feishu.streaming_card import (
    FEISHU_CARD_SAFE_BYTES,
    FeishuStreamingCardConsumer,
    _render_card_pages,
    _card_size_bytes,
)


class _FakeCardAdapter:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.updated: list[dict] = []

    async def send_streaming_card(self, chat_id, card, *, metadata=None):
        self.sent.append({"chat_id": chat_id, "card": card, "metadata": metadata})
        return SimpleNamespace(success=True, message_id="om_card")

    async def update_streaming_card(self, chat_id, message_id, card, *, finalize=False):
        self.updated.append(
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "card": card,
                "finalize": finalize,
            }
        )
        return SimpleNamespace(success=True, message_id=message_id)


class _FlakyFinalUpdateCardAdapter(_FakeCardAdapter):
    def __init__(self) -> None:
        super().__init__()
        self._failed_once = False

    async def update_streaming_card(self, chat_id, message_id, card, *, finalize=False):
        self.updated.append(
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "card": card,
                "finalize": finalize,
            }
        )
        if finalize and not self._failed_once:
            self._failed_once = True
            return SimpleNamespace(success=False, error="temporary timeout")
        return SimpleNamespace(success=True, message_id=message_id)


class _FlakyContinuationCardAdapter(_FakeCardAdapter):
    def __init__(self) -> None:
        super().__init__()
        self._next_id = 0
        self.continuation_failures = 0

    async def send_streaming_card(self, chat_id, card, *, metadata=None):
        title = card.get("header", {}).get("title", {}).get("content", "")
        if "(2/" in title and self.continuation_failures == 0:
            self.continuation_failures += 1
            return SimpleNamespace(success=False, error="temporary continuation failure")
        self._next_id += 1
        self.sent.append({"chat_id": chat_id, "card": card, "metadata": metadata})
        return SimpleNamespace(success=True, message_id=f"om_card_{self._next_id}")


@pytest.mark.asyncio
async def test_feishu_streaming_card_consumer_accumulates_turn_into_one_card():
    adapter = _FakeCardAdapter()
    consumer = FeishuStreamingCardConsumer(
        adapter,
        "oc_chat",
        metadata={"thread_id": "omt_thread", "reply_to_message_id": "om_root"},
        initial_reply_to_id="om_root",
        session_id="session-1",
    )

    task = asyncio.create_task(consumer.run())
    await asyncio.sleep(0.05)

    consumer.on_commentary("我先查一下")
    consumer.on_tool_event("tool.started", tool_name="terminal", preview="pwd")
    consumer.on_delta("最终")
    consumer.on_delta("答案")
    consumer.finish(
        "最终答案",
        duration=3.2,
        model="gpt-test",
        tokens={"input_tokens": 1000, "output_tokens": 25},
        context={"used_tokens": 2000, "max_tokens": 8000},
    )

    await asyncio.wait_for(task, timeout=2)

    assert adapter.sent
    assert adapter.sent[0]["card"]["header"]["subtitle"]["content"] == "思考中"
    final = adapter.updated[-1]["card"]
    assert adapter.updated[-1]["finalize"] is True
    assert final["header"]["template"] == "green"
    assert final["header"]["subtitle"]["content"] == "已完成"
    body_text = json.dumps(final["body"], ensure_ascii=False)
    assert "最终答案" in body_text
    assert "工具调用 1 次" in body_text
    assert "gpt-test" in body_text
    assert consumer.final_response_sent is True
    assert consumer.final_content_delivered is True
    assert consumer.message_id == "om_card"


def test_feishu_streaming_card_keeps_more_than_five_markdown_tables():
    tables = []
    for index in range(7):
        tables.append(
            f"| 项目 | 值 |\n| --- | --- |\n| table-{index} | value-{index} |"
        )
    session = streaming_card_module.CardSession(
        conversation_id="session-1",
        message_id="message-1",
        chat_id="oc_chat",
        status="completed",
        answer_text="\n\n".join(tables),
    )

    cards = _render_card_pages(session)
    body_text = json.dumps([card["body"] for card in cards], ensure_ascii=False)

    assert "超出部分已省略" not in body_text
    assert "table-0" in body_text
    assert "table-6" in body_text


@pytest.mark.asyncio
async def test_feishu_streaming_card_sends_long_final_answer_as_continuation_cards():
    adapter = _FakeCardAdapter()
    consumer = FeishuStreamingCardConsumer(adapter, "oc_chat", session_id="session-1")
    long_answer = "\n\n".join(
        f"## Section {index}\n" + ("这是一段较长的中文内容。" * 80)
        for index in range(80)
    )

    task = asyncio.create_task(consumer.run())
    await asyncio.sleep(0.05)
    consumer.finish(long_answer, duration=2.0, model="gpt-test")
    await asyncio.wait_for(task, timeout=2)

    final_cards = [adapter.updated[-1]["card"], *(call["card"] for call in adapter.sent[1:])]
    assert len(final_cards) > 1
    assert all(_card_size_bytes(card) <= FEISHU_CARD_SAFE_BYTES for card in final_cards)
    body_text = json.dumps([card["body"] for card in final_cards], ensure_ascii=False)
    assert "Section 0" in body_text
    assert "Section 79" in body_text
    assert "超出部分已省略" not in body_text


@pytest.mark.asyncio
async def test_feishu_streaming_card_retries_failed_continuation_cards(monkeypatch):
    monkeypatch.setattr(streaming_card_module, "TERMINAL_UPDATE_RETRY_DELAYS", (0.01,))
    adapter = _FlakyContinuationCardAdapter()
    consumer = FeishuStreamingCardConsumer(adapter, "oc_chat", session_id="session-1")
    long_answer = "\n\n".join(
        f"## Section {index}\n" + ("这是一段较长的中文内容。" * 80)
        for index in range(80)
    )

    task = asyncio.create_task(consumer.run())
    await asyncio.sleep(0.05)
    consumer.finish(long_answer, duration=2.0, model="gpt-test")
    await asyncio.wait_for(task, timeout=2)
    await asyncio.sleep(0.05)

    assert adapter.continuation_failures == 1
    continuation_cards = [call["card"] for call in adapter.sent[1:]]
    assert continuation_cards
    body_text = json.dumps([card["body"] for card in continuation_cards], ensure_ascii=False)
    assert "Section 79" in body_text


@pytest.mark.asyncio
async def test_feishu_streaming_card_final_update_failure_is_card_delivery_accepted(monkeypatch):
    monkeypatch.setattr(streaming_card_module, "TERMINAL_UPDATE_RETRY_DELAYS", (0.01,))
    adapter = _FlakyFinalUpdateCardAdapter()
    consumer = FeishuStreamingCardConsumer(adapter, "oc_chat", session_id="session-1")

    task = asyncio.create_task(consumer.run())
    await asyncio.sleep(0.05)

    consumer.on_commentary("我先查一下")
    consumer.finish("最终答案", duration=1.0, model="gpt-test")

    await asyncio.wait_for(task, timeout=2)
    await asyncio.sleep(0.05)

    assert consumer.final_response_sent is True
    assert consumer.final_content_delivered is True
    assert len(adapter.updated) >= 2
    assert all(call["finalize"] is True for call in adapter.updated)
    assert adapter.updated[-1]["card"]["header"]["subtitle"]["content"] == "已完成"


@pytest.mark.asyncio
async def test_feishu_adapter_sends_and_updates_streaming_card_as_interactive():
    adapter = FeishuAdapter(PlatformConfig(enabled=True))
    adapter._client = MagicMock()
    send_response = SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id="om_stream"))
    update_response = SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id="ignored"))

    with (
        patch.object(adapter, "_feishu_send_with_retry", new_callable=AsyncMock, return_value=send_response) as send_mock,
        patch.object(adapter, "_run_blocking", new_callable=AsyncMock, return_value=update_response) as run_blocking,
    ):
        sent = await adapter.send_streaming_card(
            "oc_chat",
            {"schema": "2.0", "body": {"elements": []}},
            metadata={"reply_to_message_id": "om_parent"},
        )
        updated = await adapter.update_streaming_card(
            "oc_chat",
            "om_stream",
            {"schema": "2.0", "body": {"elements": [{"tag": "markdown", "content": "done"}]}},
        )

    assert sent.success is True
    assert sent.message_id == "om_stream"
    send_kwargs = send_mock.call_args.kwargs
    assert send_kwargs["msg_type"] == "interactive"
    assert send_kwargs["reply_to"] == "om_parent"

    assert updated.success is True
    assert updated.message_id == "om_stream"
    request = run_blocking.call_args.args[1]
    assert request.http_method == "PATCH"
    assert request.uri == "/open-apis/im/v1/messages/om_stream"
    assert "msg_type" not in request.body
    assert "done" in request.body["content"]
