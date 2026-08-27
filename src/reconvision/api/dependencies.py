"""Wiring the web layer to the same adapters the pipeline uses.

Built once at startup and handed to request handlers, so a request never pays to
load a 360 MB model and two processes never disagree about what is enrolled.
"""

from __future__ import annotations

from dataclasses import dataclass

from reconvision.adapters.faces.insightface_analyzer import InsightFaceAnalyzer
from reconvision.adapters.storage.snapshots import FileSnapshotStore
from reconvision.adapters.storage.sqlite_events import SqliteEvents
from reconvision.adapters.storage.sqlite_gallery import SqliteGallery, connect
from reconvision.application.assembly import SystemClock, build_enrollment_service
from reconvision.application.config import Settings
from reconvision.application.enrollment import EnrollmentService
from reconvision.application.feedback import FeedbackService
from reconvision.domain.quality import QualityPolicy


@dataclass(frozen=True, slots=True)
class Services:
    """Everything the screens need, assembled once."""

    settings: Settings
    gallery: SqliteGallery
    events: SqliteEvents
    snapshots: FileSnapshotStore
    enrollment: EnrollmentService
    feedback: FeedbackService

    def close(self) -> None:
        self.gallery.close()


def build_services(settings: Settings) -> Services:
    """Open the database and load the models the screens depend on."""
    connection = connect(settings.database_path)
    gallery = SqliteGallery(connection)
    events = SqliteEvents(connection)
    snapshots = FileSnapshotStore(settings.snapshots_dir)
    analyzer = InsightFaceAnalyzer(settings.models_dir, settings.face_model_pack)
    quality = QualityPolicy(min_pixel_height=settings.min_face_pixels)

    enrollment = build_enrollment_service(settings, gallery=gallery, analyzer=analyzer)

    return Services(
        settings=settings,
        gallery=gallery,
        events=events,
        snapshots=snapshots,
        enrollment=enrollment,
        feedback=FeedbackService(
            events=events,
            snapshots=snapshots,
            analyzer=analyzer,
            enrollment=enrollment,
            clock=SystemClock(),
            quality=quality,
        ),
    )
