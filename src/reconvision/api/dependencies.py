"""Wiring the web layer to the same adapters the pipeline uses.

Built once at startup and handed to request handlers, so a request never pays to
load 360 MB of weights and two processes never disagree about what is enrolled.

The face models are loaded lazily and on purpose. A container started for the
first time has no weights yet, and refusing to start is the wrong response: the
review screen needs only the database, and the screen is where someone would find
out what to do about the missing models. Failing at startup would leave them with
a crash-looping container and no page to read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import structlog

from reconvision.adapters.faces.insightface_analyzer import InsightFaceAnalyzer
from reconvision.adapters.storage.snapshots import FileSnapshotStore
from reconvision.adapters.storage.sqlite_events import SqliteEvents
from reconvision.adapters.storage.sqlite_gallery import SqliteGallery, connect
from reconvision.application.assembly import SystemClock
from reconvision.application.config import Settings
from reconvision.application.enrollment import EnrollmentService
from reconvision.application.feedback import FeedbackService
from reconvision.domain.models import BoundingBox, Face, Frame
from reconvision.domain.ports import FaceAnalyzer
from reconvision.domain.quality import QualityPolicy

logger = structlog.get_logger(__name__)

MODELS_MISSING_MESSAGE = (
    "Face models are not downloaded yet. Run `reconvision export-models` "
    "(in the container: `docker compose exec reconvision reconvision export-models`)."
)


class ModelsUnavailableError(RuntimeError):
    """A screen needed the face models and they are not present."""

    def __init__(self) -> None:
        super().__init__(MODELS_MISSING_MESSAGE)


class LazyFaceAnalyzer:
    """Loads the face models on first use, not at startup.

    Satisfies the `FaceAnalyzer` port, so nothing downstream knows the difference
    between this and a loaded analyzer - until the models are genuinely needed and
    genuinely absent, at which point it says so in a sentence a person can act on.
    """

    def __init__(self, models_dir: Path, pack: str) -> None:
        self._models_dir = models_dir
        self._pack = pack
        self._loaded: InsightFaceAnalyzer | None = None

    @property
    def is_available(self) -> bool:
        """Whether the weights are on disk, without loading them."""
        return (self._models_dir / "models" / self._pack).is_dir()

    def analyse(self, frame: Frame, region: BoundingBox | None = None) -> list[Face]:
        if self._loaded is None:
            try:
                self._loaded = InsightFaceAnalyzer(self._models_dir, self._pack)
            except FileNotFoundError as error:
                raise ModelsUnavailableError from error
        return list(self._loaded.analyse(frame, region))


@dataclass(frozen=True, slots=True)
class Services:
    """Everything the screens need, assembled once."""

    settings: Settings
    gallery: SqliteGallery
    events: SqliteEvents
    snapshots: FileSnapshotStore
    enrollment: EnrollmentService
    feedback: FeedbackService
    analyzer: FaceAnalyzer | None = field(default=None)

    @property
    def models_available(self) -> bool:
        """Whether the screens that need face models can do their work."""
        analyzer = self.analyzer
        return not isinstance(analyzer, LazyFaceAnalyzer) or analyzer.is_available

    def close(self) -> None:
        self.gallery.close()


def build_services(settings: Settings) -> Services:
    """Open the database and prepare the models the screens depend on."""
    connection = connect(settings.database_path)
    gallery = SqliteGallery(connection)
    events = SqliteEvents(connection)
    snapshots = FileSnapshotStore(settings.snapshots_dir)
    analyzer = LazyFaceAnalyzer(settings.models_dir, settings.face_model_pack)
    quality = QualityPolicy(min_pixel_height=settings.min_face_pixels)
    clock = SystemClock()

    if not analyzer.is_available:
        logger.warning("models_absent", models_dir=str(settings.models_dir))

    enrollment = EnrollmentService(analyzer=analyzer, gallery=gallery, clock=clock, quality=quality)

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
            clock=clock,
            quality=quality,
        ),
        analyzer=analyzer,
    )
