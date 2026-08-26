"""Wiring the concrete adapters into a pipeline.

The one place that knows which implementation satisfies which port. Everything
else depends on the protocols in `domain.ports`, which is what lets the tests
substitute fakes and what keeps a change of tracker or detector local.
"""

from __future__ import annotations

from datetime import UTC, datetime

from reconvision.adapters.detection.onnx_yolo import DetectorPolicy, OnnxYoloDetector
from reconvision.adapters.faces.insightface_analyzer import InsightFaceAnalyzer
from reconvision.adapters.storage.sqlite_gallery import SqliteGallery, connect
from reconvision.adapters.tracking.bytetrack import ByteTrackAdapter, TrackingPolicy
from reconvision.adapters.video.sources import create_frame_source
from reconvision.application.config import CameraConfig, Settings
from reconvision.application.enrollment import EnrollmentService
from reconvision.application.ingest import FrameIngestor
from reconvision.application.pipeline import RecognitionPipeline
from reconvision.application.telemetry import Telemetry
from reconvision.domain.matching import GalleryMatcher, ThresholdPolicy
from reconvision.domain.ports import GalleryRepository
from reconvision.domain.quality import QualityPolicy

#: Detector filename produced by `reconvision export-models`.
DETECTOR_FILENAME = "yolo11s.onnx"


class SystemClock:
    """Wall time, in UTC.

    UTC throughout: events are compared and retained across daylight-saving
    changes, and a duplicate or missing hour in the record is not worth the
    convenience of local time.
    """

    def now(self) -> datetime:
        return datetime.now(UTC)


def build_matcher(gallery: GalleryRepository, settings: Settings) -> GalleryMatcher:
    """Load the enrolled gallery into a matcher."""
    return GalleryMatcher(
        entries=gallery.load_entries(),
        policy=ThresholdPolicy(
            match_threshold=settings.match_threshold,
            min_margin=settings.min_match_margin,
        ),
    )


def build_pipeline(
    camera: CameraConfig,
    settings: Settings,
    gallery: GalleryRepository,
    telemetry: Telemetry,
    detector: OnnxYoloDetector | None = None,
    analyzer: InsightFaceAnalyzer | None = None,
) -> RecognitionPipeline:
    """Assemble everything needed to watch one camera.

    The detector and analyzer are injectable so several cameras can share one
    loaded copy of the weights: they are around 360 MB, and loading them per
    camera is what would actually exhaust a NAS.
    """
    source = create_frame_source(camera.source, name=camera.name)

    return RecognitionPipeline(
        camera_name=camera.name,
        ingestor=FrameIngestor(
            source=source,
            telemetry=telemetry,
            sample_every_n_frames=camera.sample_every_n_frames,
        ),
        detector=detector
        or OnnxYoloDetector(settings.models_dir / DETECTOR_FILENAME, DetectorPolicy()),
        analyzer=analyzer or InsightFaceAnalyzer(settings.models_dir, settings.face_model_pack),
        tracker=ByteTrackAdapter(TrackingPolicy()),
        matcher=build_matcher(gallery, settings),
        clock=SystemClock(),
        telemetry=telemetry,
        quality_policy=QualityPolicy(min_pixel_height=settings.min_face_pixels),
    )


def open_gallery(settings: Settings) -> SqliteGallery:
    """Open the durable gallery, creating and migrating the database if needed."""
    return SqliteGallery(connect(settings.database_path))


def build_enrollment_service(
    settings: Settings,
    gallery: GalleryRepository | None = None,
    analyzer: InsightFaceAnalyzer | None = None,
) -> EnrollmentService:
    """Assemble enrolment. Needs the face models but not the detector."""
    return EnrollmentService(
        analyzer=analyzer or InsightFaceAnalyzer(settings.models_dir, settings.face_model_pack),
        gallery=gallery if gallery is not None else open_gallery(settings),
        clock=SystemClock(),
        quality=QualityPolicy(min_pixel_height=settings.min_face_pixels),
    )
