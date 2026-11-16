import pytest

from src.no_helmet_detection.events import EventLedger


def test_violation_is_reported_after_persistent_evidence_and_plate_reads_are_merged() -> None:
    ledger = EventLedger(min_frames=3, persistence_gap=1)

    assert ledger.update(8, 10, subject_type="person").evidence_frames == 1
    ledger.update(8, 11, "ab12 cd3456", 0.8, subject_type="person")
    event = ledger.update(8, 13, "AB12CD3456", 0.9, subject_type="person")

    assert event.evidence_frames == 3
    assert event.plate_text == "AB12CD3456"
    assert event.plate_confidence == pytest.approx(0.85)
    assert ledger.confirmed_events == [event]


def test_long_gap_starts_a_new_unconfirmed_event() -> None:
    ledger = EventLedger(min_frames=2, persistence_gap=1)
    first = ledger.update(5, 1)
    second = ledger.update(5, 5)

    assert first is not second
    assert first.evidence_frames == 1
    assert second.evidence_frames == 1
    assert ledger.confirmed_events == []


def test_person_and_motorcycle_ids_do_not_share_an_event() -> None:
    ledger = EventLedger(min_frames=1, persistence_gap=1)

    person = ledger.update(4, 1, subject_type="person")
    motorcycle = ledger.update(4, 1, subject_type="motorcycle")

    assert person is not motorcycle
    assert person.subject_type == "person"
    assert motorcycle.subject_type == "motorcycle"
