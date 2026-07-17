from datetime import datetime, timezone

from cdm_archive_provider import _query_path, prepare_archive_row


def test_query_uses_public_cdm_created_axis():
    start = datetime(2026, 7, 16, tzinfo=timezone.utc)
    end = datetime(2026, 7, 17, tzinfo=timezone.utc)
    path = _query_path(start, end, 50000)
    assert path.startswith("/class/cdm_public/CREATED/")
    assert "/orderby/CREATED%20asc/" in path
    assert path.endswith("/limit/50000")


def test_prepare_archive_row_maps_public_schema_and_preserves_raw_fields():
    row = {
        "CDM_ID": "123",
        "CREATED": "2026-07-17T00:00:00",
        "TCA": "2026-07-18T00:00:00",
        "SAT_1_ID": "25544",
        "SAT_2_ID": "99999",
        "SAT_1_NAME": "ISS",
        "SAT_2_NAME": "TEST DEB",
        "SAT2_OBJECT_TYPE": "DEBRIS",
        "MIN_RNG": "0.450",
        "PC": "1.2e-4",
        "CUSTOM_PROVIDER_FIELD": "kept",
    }

    prepared = prepare_archive_row(row)

    assert prepared["target_norad"] == "25544"
    assert prepared["secondary_norad"] == "99999"
    assert prepared["creation_date"] == row["CREATED"]
    assert prepared["miss_distance_km"] == row["MIN_RNG"]
    assert prepared["pc"] == row["PC"]
    assert prepared["object_type"] == "DEBRIS"
    assert prepared["CUSTOM_PROVIDER_FIELD"] == "kept"


def test_prepare_archive_row_rejects_unusable_rows():
    assert prepare_archive_row({"CDM_ID": "1"}) is None
