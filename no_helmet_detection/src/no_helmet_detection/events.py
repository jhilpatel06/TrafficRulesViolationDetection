from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ViolationEvent:
    subject_track_id: int
    subject_type: str
    first_frame: int
    last_frame: int
    evidence_frames: int = 0
    plate_votes: dict[str, list[float]] = field(default_factory=dict)

    def add_plate_read(self, text: str, confidence: float) -> None:
        normalized = "".join(character for character in text.upper() if character.isalnum())
        if len(normalized) >= 3 and confidence > 0:
            self.plate_votes.setdefault(normalized, []).append(confidence)

    @property
    def plate_text(self) -> str:
        if not self.plate_votes:
            return ""
        return max(
            self.plate_votes,
            key=lambda text: (sum(self.plate_votes[text]), len(self.plate_votes[text])),
        )

    @property
    def plate_confidence(self) -> float:
        scores = self.plate_votes.get(self.plate_text, [])
        return sum(scores) / len(scores) if scores else 0.0

    def as_dict(self) -> dict[str, int | float | str]:
        return {
            "subject_track_id": self.subject_track_id,
            "subject_type": self.subject_type,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "evidence_frames": self.evidence_frames,
            "plate_text": self.plate_text,
            "plate_confidence": round(self.plate_confidence, 4),
        }


class EventLedger:
    def __init__(self, min_frames: int, persistence_gap: int) -> None:
        if min_frames < 1:
            raise ValueError("min_frames must be at least 1")
        if persistence_gap < 0:
            raise ValueError("persistence_gap cannot be negative")
        self.min_frames = min_frames
        self.persistence_gap = persistence_gap
        self.events: list[ViolationEvent] = []
        self._active: dict[tuple[str, int], ViolationEvent] = {}

    def update(
        self,
        subject_track_id: int,
        frame_number: int,
        plate_text: str = "",
        plate_confidence: float = 0.0,
        subject_type: str = "motorcycle",
    ) -> ViolationEvent:
        subject_key = (subject_type, subject_track_id)
        event = self._active.get(subject_key)
        if event is None or frame_number - event.last_frame > self.persistence_gap + 1:
            event = ViolationEvent(
                subject_track_id=subject_track_id,
                subject_type=subject_type,
                first_frame=frame_number,
                last_frame=frame_number,
            )
            self._active[subject_key] = event
            self.events.append(event)

        event.last_frame = frame_number
        event.evidence_frames += 1
        event.add_plate_read(plate_text, plate_confidence)
        return event

    @property
    def confirmed_events(self) -> list[ViolationEvent]:
        return [event for event in self.events if event.evidence_frames >= self.min_frames]
