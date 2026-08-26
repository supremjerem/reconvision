"""The ports are contracts, so what matters is that they can actually be met.

These assertions catch the drift that would otherwise surface only when a real
adapter is wired in: a port gaining a method no implementation provides, or a
signature changing under the doubles the rest of the suite is built on.

Each test does two things. The annotated binding makes mypy verify the *signatures*
match structurally, which is the real contract; the isinstance call verifies the
same at runtime, which is all `runtime_checkable` can see.
"""

from __future__ import annotations

from reconvision.domain.ports import (
    Clock,
    EventRepository,
    FrameSource,
    GalleryRepository,
    Notifier,
    ObjectDetector,
)
from tests.fakes import (
    FakeClock,
    InMemoryEventRepository,
    InMemoryGalleryRepository,
    RecordingNotifier,
    ScriptedDetector,
    ScriptedFrameSource,
)


def test_the_gallery_port_is_satisfiable() -> None:
    gallery: GalleryRepository = InMemoryGalleryRepository()

    assert isinstance(gallery, GalleryRepository)


def test_the_event_port_is_satisfiable() -> None:
    events: EventRepository = InMemoryEventRepository()

    assert isinstance(events, EventRepository)


def test_the_notifier_port_is_satisfiable() -> None:
    notifier: Notifier = RecordingNotifier()

    assert isinstance(notifier, Notifier)


def test_the_frame_source_port_is_satisfiable() -> None:
    source: FrameSource = ScriptedFrameSource("webcam", [])

    assert isinstance(source, FrameSource)


def test_the_detector_port_is_satisfiable() -> None:
    detector: ObjectDetector = ScriptedDetector([])

    assert isinstance(detector, ObjectDetector)


def test_the_clock_port_is_satisfiable() -> None:
    clock: Clock = FakeClock()

    assert isinstance(clock, Clock)


def test_an_unrelated_object_does_not_accidentally_satisfy_a_port() -> None:
    """Guards against a port so loose that anything passes it."""
    assert not isinstance(object(), GalleryRepository)
