from datetime import datetime, timedelta, timezone

from cdm_auto_sync import _calculate_window, _json_rows, _query_path


def test_json_rows_accepts_list_and_object():
    assert _json_rows('[{"CDM_ID": "1"}]') == [{"CDM_ID": "1"}]
    assert _json_rows('{"CDM_ID": "1"}') == [{"CDM_ID": "1"}]


def test_initial_window_is_limited_to_retention():
    now = datetime(2026, 7, 17, 12, 0, tzinfo=timezone.utc)
    start, end, gap = _calculate_window(
        state={},
        now=now,
        initial_lookback_days=90,
        overlap_hours=24,
        max_lookback_days=30,
    )
    assert start == now - timedelta(days=30)
    assert end == now
    assert gap == 0.0


def test_incremental_window_uses_overlap():
    now = datetime(2026, 7, 17, 12, 0, tzinfo=timezone.utc)
    last_success = now - timedelta(hours=8)
    start, _, gap = _calculate_window(
        state={"last_success_at": last_success.isoformat()},
        now=now,
        initial_lookback_days=30,
        overlap_hours=24,
        max_lookback_days=30,
    )
    assert start == last_success - timedelta(hours=24)
    assert gap == 0.0


def test_query_is_incremental_and_bounded():
    start = datetime(2026, 7, 16, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 7, 17, 0, 0, tzinfo=timezone.utc)
    path = _query_path(start, end, 50000)
    assert path.startswith("/class/cdm_public/CREATION_DATE/")
    assert "/orderby/CREATION_DATE%20asc/" in path
    assert path.endswith("/limit/50000")
