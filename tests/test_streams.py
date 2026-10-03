"""One stream reader for every server: wherever the reasoning goes, it is found."""

from __future__ import annotations

import json

from flyball.streams import InlineSplitter, SSEReader, read_message


def event(delta: dict, finish: str | None = None, cid: str = "c1") -> bytes:
    choice: dict = {"delta": delta}
    if finish:
        choice["finish_reason"] = finish
    return b"data: " + json.dumps({"id": cid, "choices": [choice]}).encode() + b"\n\n"


def read(chunks: list[bytes], reader: SSEReader | None = None, size: int = 0):
    reader = reader or SSEReader()
    data = b"".join(chunks)
    pieces = [data[i:i + size] for i in range(0, len(data), size)] if size else chunks
    deltas = [d for piece in pieces for d in reader.feed(piece)]
    return reader, "".join(d.reasoning for d in deltas), "".join(d.content for d in deltas), deltas


def test_every_spelling_of_the_reasoning_is_read() -> None:
    for key in ("reasoning_content", "reasoning", "thinking"):
        _, reasoning, content, _ = read([event({key: "hmm"}), event({"content": "42"})])
        assert (reasoning, content) == ("hmm", "42"), key


def test_a_server_that_sends_two_spellings_is_counted_once() -> None:
    _, reasoning, _, _ = read([event({"reasoning_content": "abc", "reasoning": "abc"})])
    assert reasoning == "abc"


def test_lines_split_across_chunks_are_reassembled() -> None:
    chunks = [event({"reasoning_content": "x" * 50}) for _ in range(4)] + [event({"content": "ok"}, "stop")]
    reader, reasoning, content, deltas = read(chunks + [b"data: [DONE]\n\n"], size=17)
    assert len(reasoning) == 200 and content == "ok"
    assert deltas[-1].finish_reason == "stop"
    assert reader.done and reader.completion_id == "c1"


def test_garbage_is_skipped() -> None:
    _, reasoning, _, _ = read([b"data: {not json\n\n", b": keep-alive\n\n", b"data: [1, 2]\n\n",
                               event({"reasoning_content": "fine"})])
    assert reasoning == "fine"


def test_tool_calls_are_counted() -> None:
    _, _, _, deltas = read([event({"tool_calls": [{"index": 0}, {"index": 1}]})])
    assert deltas[0].tool_calls == 2


def test_inline_reasoning_is_split_from_the_reply_even_when_tags_are_cut() -> None:
    chunks = [event({"content": piece}) for piece in ["<thi", "nk>first ", "thought</th", "ink>\n\nthe reply"]]
    reader = SSEReader(InlineSplitter())
    _, reasoning, content, _ = read(chunks, reader)
    assert reasoning == "first thought"
    assert content == "\n\nthe reply"


def test_a_template_that_opens_the_reasoning_streams_only_its_end() -> None:
    reader = SSEReader(InlineSplitter(opened=True))
    _, reasoning, content, _ = read([event({"content": "thinking</think>answer"})], reader)
    assert (reasoning, content) == ("thinking", "answer")


def test_text_that_only_looks_like_a_tag_is_kept() -> None:
    reader = SSEReader(InlineSplitter())
    _, reasoning, content, _ = read([event({"content": "a <thin"}), event({"content": "g> here"})], reader)
    assert (reasoning, content) == ("", "a <thing> here")


def test_a_response_that_was_not_streamed_is_read_the_same_way() -> None:
    body = json.dumps({"choices": [{"message": {"reasoning": "r", "content": "hi"}, "finish_reason": "stop"}]})
    delta = read_message(body.encode())
    assert (delta.reasoning, delta.content, delta.finish_reason) == ("r", "hi", "stop")
    assert read_message(b"not json") is None
