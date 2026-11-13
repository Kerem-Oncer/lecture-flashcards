"""Tests for topic planning and coverage. A fake _call_claude means no API key or network."""

import json
import threading
import time
from types import SimpleNamespace

import pytest

import generator
from generator import (
    CHUNK_CHARS,
    INPUT_PRICE_PER_M,
    MAX_CARDS,
    OUTPUT_PRICE_PER_M,
    generate_cards,
)

FAKE_INPUT_TOKENS = 1000
FAKE_OUTPUT_TOKENS = 200


def card(topic: str, front: str | None = None) -> dict:
    return {
        "front": front or f"Question about {topic}?",
        "back": f"Answer about {topic}.",
        "card_type": "concept",
        "source_quote": topic,
        "tags": [],
        "topic": topic,
    }


def reply(topics: list[str], cards: list[dict]) -> str:
    return json.dumps({"topics": topics, "cards": cards})


def is_follow_up(messages: list[dict]) -> bool:
    return "These topics have no card yet" in messages[-1]["content"]


class FakeClaude:
    """Stands in for _call_claude. handler(messages) returns the reply text.

    Returns the text with a fake usage object, like the real response.usage.
    Chunks run in parallel threads, so recording calls uses a lock.
    """

    def __init__(self, handler):
        self.handler = handler
        self.calls: list[list[dict]] = []
        self._lock = threading.Lock()

    def __call__(self, client, messages):
        with self._lock:
            self.calls.append(list(messages))  # Copy: the caller appends to its list later.
        usage = SimpleNamespace(input_tokens=FAKE_INPUT_TOKENS, output_tokens=FAKE_OUTPUT_TOKENS)
        return self.handler(messages), usage


@pytest.fixture
def fake_claude(monkeypatch):
    monkeypatch.setattr(generator, "_get_api_key", lambda: "test-key")

    def install(handler) -> FakeClaude:
        fake = FakeClaude(handler)
        monkeypatch.setattr(generator, "_call_claude", fake)
        return fake

    return install


def two_chunk_text() -> str:
    # Each paragraph is about 2,000 characters, so together they split into 2 chunks.
    return "CHUNK_ONE " + "word " * 400 + "\n\n" + "CHUNK_TWO " + "word " * 400


def test_card_with_unlisted_topic_fails_validation_and_retries(fake_claude):
    def handler(messages):
        if len(messages) == 1:
            return reply(["TCP"], [card("TCP"), card("Bogus")])
        return reply(["TCP"], [card("TCP")])

    fake = fake_claude(handler)
    result = generate_cards("TCP is a protocol.", 5, "intro", "", auto=True)

    assert len(fake.calls) == 2
    retry_message = fake.calls[1][-1]["content"]
    assert "failed validation" in retry_message
    assert "Bogus" in retry_message
    assert "minified JSON" in retry_message
    assert [c.topic for c in result.cards] == ["TCP"]


def test_missing_topics_trigger_one_follow_up_and_cards_are_added(fake_claude):
    topics = ["Mitosis", "Meiosis", "Osmosis"]

    def handler(messages):
        if is_follow_up(messages):
            return json.dumps({"cards": [card("Meiosis"), card("Osmosis")]})
        return reply(topics, [card("Mitosis")])

    fake = fake_claude(handler)
    result = generate_cards("Cell biology notes.", 5, "intro", "", auto=True)

    assert len(fake.calls) == 2
    follow_up = fake.calls[1][-1]["content"]
    assert "Meiosis" in follow_up and "Osmosis" in follow_up and "Mitosis" not in follow_up
    assert "minified JSON" in follow_up and '{"cards":[...]}' in follow_up
    assert [c.topic for c in result.cards] == topics
    assert result.covered == set(topics)


def test_failed_follow_up_keeps_existing_cards_without_retrying(fake_claude):
    def handler(messages):
        if is_follow_up(messages):
            return json.dumps({"cards": [card("Not a missing topic")]})
        return reply(["Mitosis", "Meiosis"], [card("Mitosis")])

    fake = fake_claude(handler)
    result = generate_cards("Cell biology notes.", 5, "intro", "", auto=True)

    assert len(fake.calls) == 2
    assert [c.topic for c in result.cards] == ["Mitosis"]
    assert result.covered == {"Mitosis"}


def test_custom_mode_with_fewer_cards_than_topics_skips_follow_up(fake_claude):
    topics = [f"Topic {i}" for i in range(8)]

    def handler(messages):
        return reply(topics, [card(topic) for topic in topics[:5]])

    fake = fake_claude(handler)
    result = generate_cards("Short notes.", 5, "intro", "", auto=False)

    assert len(fake.calls) == 1
    assert len(result.cards) == 5
    assert result.topics == topics
    assert set(topics) - result.covered == {"Topic 5", "Topic 6", "Topic 7"}


def test_auto_mode_caps_cards_at_max_across_two_chunks(fake_claude):
    def handler(messages):
        prefix = "A" if "CHUNK_ONE" in messages[0]["content"] else "B"
        topics = [f"{prefix} topic {i}" for i in range(30)]
        if is_follow_up(messages):
            return json.dumps({"cards": [card(topic) for topic in topics[15:]]})
        return reply(topics, [card(topic) for topic in topics[:15]])

    fake = fake_claude(handler)
    result = generate_cards(two_chunk_text(), 5, "intro", "", auto=True)

    assert len(generator.chunk_text(two_chunk_text())) == 2
    assert len(fake.calls) == 4  # One request and one follow-up per chunk.
    assert len(result.cards) == MAX_CARDS


def test_topics_merge_and_dedupe_across_chunks_in_document_order(fake_claude):
    def handler(messages):
        if "CHUNK_ONE" in messages[0]["content"]:
            time.sleep(0.2)  # Finish after chunk two, to prove order isn't completion order.
            return reply(["TCP", "UDP"], [card("TCP"), card("UDP")])
        return reply(["tcp.", "IP"], [card("tcp.", "How does TCP recover from loss?"), card("IP")])

    fake_claude(handler)
    result = generate_cards(two_chunk_text(), 10, "intro", "", auto=True)

    assert result.topics == ["TCP", "UDP", "IP"]
    assert result.covered == {"TCP", "UDP", "IP"}


def test_chunks_are_at_most_3000_characters():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 100 for i in range(20))  # ~10,000 chars

    chunks = generator.chunk_text(text)

    assert CHUNK_CHARS == 3000
    assert len(chunks) == 4
    assert all(len(chunk) < 1.25 * CHUNK_CHARS for chunk in chunks)
    assert within_25_percent(chunks)


def test_tags_are_built_in_code_from_course_and_topic(fake_claude):
    fake_claude(lambda messages: reply(["French Revolution"], [card("French Revolution")]))

    with_course = generate_cards("Notes.", 5, "intro", "HST 102", auto=True)
    without_course = generate_cards("Notes.", 5, "intro", "", auto=True)

    assert with_course.cards[0].tags == ["HST_102", "French_Revolution"]
    assert without_course.cards[0].tags == ["French_Revolution"]


def test_usage_totals_count_every_call_by_type(fake_claude):
    def handler(messages):
        if is_follow_up(messages):
            return json.dumps({"cards": [card("Meiosis")]})
        if len(messages) == 1:
            return "not json"  # Forces a retry.
        return reply(["Mitosis", "Meiosis"], [card("Mitosis")])

    fake = fake_claude(handler)
    result = generate_cards("Cell biology notes.", 5, "intro", "", auto=True)

    assert len(fake.calls) == 3
    assert result.call_counts == {"first": 1, "retry": 1, "follow-up": 1}
    assert result.input_tokens == 3 * FAKE_INPUT_TOKENS
    assert result.output_tokens == 3 * FAKE_OUTPUT_TOKENS
    expected_cost = (
        3 * FAKE_INPUT_TOKENS * INPUT_PRICE_PER_M + 3 * FAKE_OUTPUT_TOKENS * OUTPUT_PRICE_PER_M
    ) / 1_000_000
    assert result.estimated_cost == pytest.approx(expected_cost)
    assert result.chunk_count == 1
    assert result.total_seconds >= 0
    assert [call.call_type for call in result.calls] == ["first", "retry", "follow-up"]


def test_on_progress_receives_each_chunks_cards(fake_claude):
    def handler(messages):
        prefix = "A" if "CHUNK_ONE" in messages[0]["content"] else "B"
        return reply([f"{prefix} topic"], [card(f"{prefix} topic")])

    fake_claude(handler)
    updates = []

    def on_progress(done, total, cards):
        updates.append((done, total, [c.topic for c in cards]))

    generate_cards(two_chunk_text(), 10, "intro", "", auto=True, on_progress=on_progress)

    assert updates[0] == (0, 2, [])
    assert [(done, total) for done, total, _ in updates[1:]] == [(1, 2), (2, 2)]
    assert sorted(topic for *_, topics in updates for topic in topics) == ["A topic", "B topic"]


def short_paragraphs(total_chars: int) -> str:
    """Numbered paragraphs of about 100 characters, so word order can be checked."""
    paragraphs, length, i = [], 0, 0
    while length < total_chars:
        paragraph = f"Point {i} " + " ".join(f"w{i}x{k}" for k in range(14))
        paragraphs.append(paragraph)
        length += len(paragraph) + 2
        i += 1
    return "\n\n".join(paragraphs)


def within_25_percent(chunks: list[str]) -> bool:
    lengths = [len(chunk) for chunk in chunks]
    return max(lengths) <= 1.25 * min(lengths)


def test_3400_characters_split_into_two_even_chunks():
    text = short_paragraphs(3400)

    chunks = generator.chunk_text(text)

    assert len(chunks) == 2
    assert within_25_percent(chunks)


def test_8900_characters_split_into_three_even_chunks():
    text = short_paragraphs(8900)

    chunks = generator.chunk_text(text)

    assert len(chunks) == 3
    assert within_25_percent(chunks)


def test_chunks_keep_every_word_in_order():
    text = short_paragraphs(8900)

    chunks = generator.chunk_text(text)

    assert " ".join(chunks).split() == text.split()


def test_one_huge_paragraph_still_splits_and_keeps_every_word():
    text = " ".join(f"Sentence {i} has a few words in it." for i in range(400))  # ~14,000 chars

    chunks = generator.chunk_text(text)

    assert len(chunks) == 5
    assert within_25_percent(chunks)
    assert " ".join(chunks).split() == text.split()
