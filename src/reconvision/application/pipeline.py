"""Orchestration: frames in, one event per passage out.

This is where the staged design pays off. Detection runs on the few frames the
ingestor let through; the face model runs only on the people detection found;
recognition is decided once per tracked path rather than once per frame. Each
stage narrows what the next one has to look at.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime

import structlog

from reconvision.application.ingest import FrameIngestor, IngestStats
from reconvision.application.telemetry import (
    EMBEDDING,
    FACE_DETECTION,
    MATCHING,
    OBJECT_DETECTION,
    TRACKING,
    Telemetry,
)
from reconvision.domain.events import EventVerdict, RecognitionEvent
from reconvision.domain.matching import GalleryMatcher
from reconvision.domain.models import (
    Detection,
    Face,
    Frame,
    SubjectKind,
    TrackedDetection,
)
from reconvision.domain.ports import Clock, FaceAnalyzer, ObjectDetector, Tracker
from reconvision.domain.quality import QualityPolicy
from reconvision.domain.smoothing import TrackVote, VotePolicy

logger = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ObservedEvent:
    """An event together with the frame that best shows its subject.

    They travel as one because every consumer needs both: storage writes the
    snapshot beside the row, a notification attaches it, and a correction turns it
    into a gallery entry. Keeping them in separate lookups invites them to drift.
    """

    event: RecognitionEvent
    snapshot: Frame | None


@dataclass(slots=True)
class _TrackState:
    """Evidence accumulated about one subject while it is in view."""

    track_id: int
    label: str
    kind: SubjectKind
    started_at: datetime
    last_seen_at: datetime
    last_seen_frame: int
    vote: TrackVote
    #: Frames this subject was seen in. Distinct from the vote's observation
    #: count, which only rises when a face was good enough to recognise: a person
    #: filmed from behind for a minute has many of the first and none of the second.
    frames_seen: int = 0
    usable_faces: int = 0
    #: The frame in which this subject looked best, kept as the event's snapshot.
    #: Holding the single best frame rather than a rolling buffer is what stops
    #: memory growing with how long someone lingers.
    best_frame: Frame | None = None
    best_weight: float = -1.0

    def verdict(self) -> EventVerdict:
        """Translate accumulated evidence into one of the four answers."""
        if self.kind is SubjectKind.ANIMAL:
            return EventVerdict.ANIMAL
        if self.usable_faces == 0:
            # Someone was here and no frame ever showed their face well enough to
            # try. Reported honestly rather than guessed at or silently dropped.
            return EventVerdict.UNIDENTIFIED

        decided = self.vote.verdict()
        if not decided.is_conclusive:
            return EventVerdict.UNIDENTIFIED
        return (
            EventVerdict.KNOWN_PERSON
            if decided.identity_id is not None
            else EventVerdict.UNKNOWN_PERSON
        )


@dataclass(frozen=True, slots=True)
class PipelinePolicy:
    """Timing rules for deciding a passage is over."""

    #: Analysed frames a track may be absent before its event is emitted. Long
    #: enough to survive the tracker briefly losing someone, short enough that a
    #: notification still arrives while it is useful.
    frames_before_closing: int = 15


class RecognitionPipeline:
    """Turns one camera's frames into recognition events."""

    def __init__(
        self,
        camera_name: str,
        ingestor: FrameIngestor,
        detector: ObjectDetector,
        analyzer: FaceAnalyzer,
        tracker: Tracker,
        matcher: GalleryMatcher,
        clock: Clock,
        telemetry: Telemetry,
        quality_policy: QualityPolicy | None = None,
        vote_policy: VotePolicy | None = None,
        policy: PipelinePolicy | None = None,
    ) -> None:
        self._camera_name = camera_name
        self._ingestor = ingestor
        self._detector = detector
        self._analyzer = analyzer
        self._tracker = tracker
        self._matcher = matcher
        self._clock = clock
        self._telemetry = telemetry
        self._quality = quality_policy or QualityPolicy()
        self._vote_policy = vote_policy or VotePolicy()
        self._policy = policy or PipelinePolicy()

        self._tracks: dict[int, _TrackState] = {}
        self._frame_index = 0

    @property
    def ingest_stats(self) -> IngestStats:
        """Throughput counters for the camera this pipeline is watching."""
        return self._ingestor.stats

    @property
    def analysed_frames(self) -> int:
        """Frames that reached detection, which is what bounds the CPU cost."""
        return self._frame_index

    def stop(self) -> None:
        """Ask the source to stop, so a live camera unblocks on shutdown."""
        self._ingestor.close()

    def close(self) -> None:
        """Release the video source and report what the camera cost."""
        self._ingestor.close()
        self._ingestor.log_throughput()

    def events(self) -> Iterator[ObservedEvent]:
        """Yield one event per subject that has finished passing through."""
        for frame in self._ingestor.analysable_frames():
            self._frame_index += 1
            yield from self._process_frame(frame)

        # The stream ended, so every track in flight has ended with it.
        yield from self._close_all_tracks()

    def _process_frame(self, frame: Frame) -> Iterator[ObservedEvent]:
        camera = {"camera": self._camera_name}
        now = self._clock.now()

        with self._telemetry.stage(OBJECT_DETECTION, **camera):
            detections = self._detector.detect(frame)
        for detection in detections:
            self._telemetry.metrics.detections.add(1, {**camera, "label": detection.label})

        with self._telemetry.stage(TRACKING, **camera):
            tracked = self._tracker.update(detections)

        for subject in tracked:
            self._observe(subject, frame, now)

        yield from self._close_expired_tracks()

    def _observe(self, subject: TrackedDetection, frame: Frame, now: datetime) -> None:
        """Fold one frame's view of one subject into its accumulated evidence."""
        state = self._tracks.get(subject.track_id)
        if state is None:
            state = _TrackState(
                track_id=subject.track_id,
                label=subject.detection.label,
                kind=subject.detection.kind or SubjectKind.PERSON,
                started_at=now,
                last_seen_at=now,
                last_seen_frame=self._frame_index,
                vote=TrackVote(policy=self._vote_policy),
            )
            self._tracks[subject.track_id] = state

        state.last_seen_at = now
        state.last_seen_frame = self._frame_index
        state.frames_seen += 1

        if state.kind is SubjectKind.ANIMAL:
            # The whole point of the gate: the face model is never invoked here.
            if state.best_frame is None:
                state.best_frame = frame
            return

        self._recognise(state, subject.detection, frame)

    def _recognise(self, state: _TrackState, detection: Detection, frame: Frame) -> None:
        """Look for a usable face on this person and fold the result into the vote."""
        camera = {"camera": self._camera_name}

        with self._telemetry.stage(FACE_DETECTION, **camera):
            faces = self._analyzer.analyse(frame, detection.box)

        usable = self._best_usable_face(faces, camera)
        if usable is None:
            return

        state.usable_faces += 1
        weight = usable.quality.weight()

        with self._telemetry.stage(EMBEDDING, **camera), self._telemetry.stage(MATCHING, **camera):
            match = self._matcher.match(usable.embedding)

        # An empty gallery reports -1.0 as a sentinel rather than a measurement,
        # and a histogram rejects negatives outright, which silently dropped every
        # sample. Only real comparisons are recorded, clamped to the [0, 1] range
        # a cosine similarity between faces occupies in practice.
        if not self._matcher.is_empty:
            self._telemetry.metrics.match_similarity.record(max(0.0, match.similarity), camera)
        state.vote.observe(match, weight)

        # The best frame is the one where the face was largest, sharpest and most
        # frontal - the most useful image to show a human, and the most useful
        # capture to add to the gallery if they later correct the event.
        if weight > state.best_weight:
            state.best_weight = weight
            state.best_frame = frame

    def _best_usable_face(self, faces: Sequence[Face], camera: dict[str, str]) -> Face | None:
        """Pick the most identifiable face, or None if none clears the quality gate.

        Only one face per person per frame is embedded. A second face inside the
        same person's box is either a duplicate detection or a bystander, and
        neither should get a vote on this track.
        """
        candidates: list[Face] = []
        for face in faces:
            reason = self._quality.rejection_reason(face.quality)
            if reason is None:
                candidates.append(face)
            else:
                self._telemetry.metrics.faces_rejected.add(1, {**camera, "reason": reason.value})

        return max(candidates, key=lambda face: face.quality.weight(), default=None)

    def _close_expired_tracks(self) -> Iterator[ObservedEvent]:
        """Emit events for subjects that have not been seen for a while."""
        expired = [
            track_id
            for track_id, state in self._tracks.items()
            if self._frame_index - state.last_seen_frame > self._policy.frames_before_closing
        ]
        for track_id in expired:
            yield self._close(self._tracks.pop(track_id))

    def _close_all_tracks(self) -> Iterator[ObservedEvent]:
        for state in list(self._tracks.values()):
            yield self._close(state)
        self._tracks.clear()

    def _close(self, state: _TrackState) -> ObservedEvent:
        """Turn a finished track into the single event that represents it."""
        verdict = state.verdict()
        decided = state.vote.verdict()
        identity_id = decided.identity_id if verdict is EventVerdict.KNOWN_PERSON else None

        event = RecognitionEvent(
            camera_name=self._camera_name,
            verdict=verdict,
            started_at=state.started_at,
            ended_at=state.last_seen_at,
            subject_kind=state.kind,
            identity_id=identity_id,
            confidence=decided.confidence,
            best_similarity=state.vote.best_similarity,
            observations=state.frames_seen,
            animal_label=state.label if state.kind is SubjectKind.ANIMAL else None,
        )

        self._telemetry.metrics.events_emitted.add(
            1, {"camera": self._camera_name, "verdict": verdict.value}
        )
        logger.info(
            "event",
            camera=self._camera_name,
            verdict=verdict.value,
            identity=identity_id,
            label=state.label,
            frames_seen=state.frames_seen,
            recognised_frames=state.vote.observations,
            usable_faces=state.usable_faces,
            duration_seconds=round(event.duration_seconds, 1),
        )
        return ObservedEvent(event=event, snapshot=state.best_frame)
