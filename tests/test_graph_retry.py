"""Tests for bounded retry state transitions."""

from app.graph.graph import _route_after_retry
from app.graph.nodes import build_retry_node


def test_retry_node_stops_after_two_retries() -> None:
    retry_node = build_retry_node()

    first_attempt = retry_node({"question": "Question", "retry_count": 0})
    assert first_attempt["retry_count"] == 1
    assert _route_after_retry({"question": "Question", "retry_count": 1}) == "rewrite"

    second_attempt = retry_node({"question": "Question", "retry_count": 1})
    assert second_attempt["retry_count"] == 2
    assert _route_after_retry({"question": "Question", "retry_count": 2}) == "end"
