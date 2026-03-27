"""
Unit tests for the structured logging module.

What we test:
- JSON output format (not plain text)
- Mandatory fields present in every log record
- correlation_id is bound and propagated via context
- bind_correlation_id / clear_correlation_id work correctly
- All required event names are loggable without errors
"""
import json
import io
import pytest

from app.logging.logger import get_logger, bind_correlation_id, clear_correlation_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _capture_log(level: str, event: str, **kwargs) -> dict:
    """Emit one log entry and return it parsed as a dict."""
    buf = io.StringIO()
    logger = get_logger(stream=buf)
    getattr(logger, level)(event, **kwargs)
    output = buf.getvalue().strip()
    assert output, "Logger produced no output"
    return json.loads(output)


# ---------------------------------------------------------------------------
# JSON format
# ---------------------------------------------------------------------------

class TestJsonFormat:
    def test_output_is_valid_json(self):
        record = _capture_log("info", "test_event")
        assert isinstance(record, dict)

    def test_single_line_per_event(self):
        buf = io.StringIO()
        logger = get_logger(stream=buf)
        logger.info("evt1")
        logger.info("evt2")
        lines = [l for l in buf.getvalue().splitlines() if l.strip()]
        assert len(lines) == 2


# ---------------------------------------------------------------------------
# Mandatory fields
# ---------------------------------------------------------------------------

class TestMandatoryFields:
    def test_timestamp_present(self):
        record = _capture_log("info", "test_event")
        assert "timestamp" in record

    def test_level_present(self):
        record = _capture_log("info", "test_event")
        assert "level" in record

    def test_event_present(self):
        record = _capture_log("info", "pipeline_started")
        assert record["event"] == "pipeline_started"

    def test_level_info(self):
        record = _capture_log("info", "test_event")
        assert record["level"].upper() == "INFO"

    def test_level_warning(self):
        record = _capture_log("warning", "test_event")
        assert record["level"].upper() in ("WARNING", "WARN")

    def test_level_error(self):
        record = _capture_log("error", "test_event")
        assert record["level"].upper() == "ERROR"

    def test_extra_kwargs_included(self):
        record = _capture_log("info", "compliance_analyzed", pessoa_id=1, duration_ms=42)
        assert record["pessoa_id"] == 1
        assert record["duration_ms"] == 42


# ---------------------------------------------------------------------------
# correlation_id
# ---------------------------------------------------------------------------

class TestCorrelationId:
    def setup_method(self):
        clear_correlation_id()

    def teardown_method(self):
        clear_correlation_id()

    def test_correlation_id_present_after_bind(self):
        bind_correlation_id("test-uuid-1234")
        record = _capture_log("info", "test_event")
        assert record.get("correlation_id") == "test-uuid-1234"

    def test_correlation_id_absent_when_not_bound(self):
        record = _capture_log("info", "test_event")
        # When not set, field should be absent or None — never a stale value
        assert record.get("correlation_id") is None or "correlation_id" not in record

    def test_correlation_id_cleared(self):
        bind_correlation_id("uuid-aaa")
        clear_correlation_id()
        record = _capture_log("info", "test_event")
        assert record.get("correlation_id") is None or "correlation_id" not in record

    def test_correlation_id_overwritten(self):
        bind_correlation_id("uuid-first")
        bind_correlation_id("uuid-second")
        record = _capture_log("info", "test_event")
        assert record.get("correlation_id") == "uuid-second"


# ---------------------------------------------------------------------------
# Required pipeline events (smoke test — all must log without raising)
# ---------------------------------------------------------------------------

REQUIRED_EVENTS = [
    ("info",    "pipeline_started"),
    ("info",    "vision_processed"),
    ("info",    "rules_retrieved"),
    ("info",    "compliance_analyzed"),
    ("info",    "pipeline_completed"),
    ("warning", "no_people_detected"),
    ("warning", "no_rules_found"),
    ("error",   "llm_call_failed"),
    ("error",   "pipeline_failed"),
]

class TestRequiredEvents:
    @pytest.mark.parametrize("level,event", REQUIRED_EVENTS)
    def test_event_logs_without_error(self, level, event):
        record = _capture_log(level, event)
        assert record["event"] == event
