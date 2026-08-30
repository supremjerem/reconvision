"""The two screens: reviewing enrolment, and correcting what the system decided.

Deliberately not a dashboard. Watching a live feed is what the camera's own app is
for; these screens exist to do the two things nothing else can - check that the
right face was enrolled, and tell the system when it got someone wrong.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from reconvision import __version__
from reconvision.api.dependencies import Services, build_services
from reconvision.application.config import Settings
from reconvision.application.enrollment import find_photos
from reconvision.application.feedback import UnknownEventError
from reconvision.domain.events import EventVerdict, FeedbackLabel, RecognitionEvent
from reconvision.domain.models import Identity

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"

#: Events shown on one page of the review screen.
EVENTS_PER_PAGE = 40


def create_app(settings: Settings | None = None, services: Services | None = None) -> FastAPI:
    """Build the application. Services are injectable so tests skip loading models."""
    resolved_settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Only built here when nothing was injected: loading 360 MB of weights is
        # worth deferring to startup, but an injected set is ready now, and a test
        # should not have to enter a context manager to reach it.
        if getattr(app.state, "services", None) is None:
            app.state.services = build_services(resolved_settings)
        yield
        app.state.services.close()

    app = FastAPI(
        title="ReconVision",
        version=__version__,
        lifespan=lifespan,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )
    app.state.services = services
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    templates.env.filters["local_time"] = _local_time
    templates.env.filters["scale_position"] = _scale_position

    def current(request: Request) -> Services:
        return request.app.state.services  # type: ignore[no-any-return]

    # --- screens ---------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    def review(request: Request, camera: str | None = None) -> HTMLResponse:
        """The correction screen: what happened, and a way to say it was wrong."""
        services = current(request)
        events = services.events.list_recent(limit=EVENTS_PER_PAGE, camera_name=camera)
        return templates.TemplateResponse(
            request=request,
            name="review.html",
            context={
                "active": "review",
                "events": events,
                "identities": services.gallery.list_identities(),
                "cameras": _cameras_in(events),
                "selected_camera": camera,
                "threshold": services.settings.match_threshold,
                "reviewed": {f.event_id for f in services.events.list_feedback()},
            },
        )

    @app.get("/people", response_class=HTMLResponse)
    def people(request: Request) -> HTMLResponse:
        """The enrolment screen: who is known, and how well."""
        services = current(request)
        identities = services.gallery.list_identities()
        return templates.TemplateResponse(
            request=request,
            name="people.html",
            context={
                "active": "people",
                "people": [
                    {
                        "identity": identity,
                        "entries": services.gallery.count_entries(identity.identity_id),
                    }
                    for identity in identities
                ],
            },
        )

    # --- actions ---------------------------------------------------------------

    @app.post("/events/{event_id}/feedback", response_class=HTMLResponse)
    def submit_feedback(
        request: Request,
        event_id: str,
        label: Annotated[str, Form()],
        identity_id: Annotated[str | None, Form()] = None,
    ) -> HTMLResponse:
        """Record a correction and swap the row for its result."""
        services = current(request)
        try:
            chosen = FeedbackLabel(label)
        except ValueError:
            raise HTTPException(status_code=422, detail=f"Unknown label {label!r}") from None

        try:
            outcome = services.feedback.submit(
                event_id=event_id,
                label=chosen,
                corrected_identity_id=identity_id or None,
            )
        except UnknownEventError:
            raise HTTPException(status_code=404, detail="No such event") from None
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from None

        return templates.TemplateResponse(
            request=request,
            name="partials/feedback_result.html",
            context={"outcome": outcome},
        )

    @app.post("/people", response_class=HTMLResponse)
    async def enroll_person(
        request: Request,
        identity_id: Annotated[str, Form()],
        photos: Annotated[list[UploadFile], File()],
        display_name: Annotated[str, Form()] = "",
    ) -> HTMLResponse:
        """Inspect uploaded photographs and enrol the usable ones."""
        services = current(request)
        identity = Identity(
            identity_id=identity_id.strip().lower().replace(" ", "_"),
            display_name=display_name.strip() or identity_id,
        )

        with TemporaryDirectory() as staging:
            folder = Path(staging)
            for upload in photos:
                if upload.filename:
                    (folder / Path(upload.filename).name).write_bytes(await upload.read())

            found = find_photos(folder)
            results = list(services.enrollment.inspect(found))
            report = services.enrollment.enroll(identity, found, results)

        return templates.TemplateResponse(
            request=request,
            name="partials/enrollment_report.html",
            context={"report": report},
        )

    @app.post("/people/{identity_id}/forget", response_class=HTMLResponse)
    def forget_person(request: Request, identity_id: str) -> HTMLResponse:
        """Delete a person's face data and unname their past events."""
        from reconvision.application.assembly import forget_identity

        services = current(request)
        changed = forget_identity(services.gallery, services.events, identity_id)
        return HTMLResponse(
            f'<p class="note">Forgotten. {changed} past event(s) no longer name them.</p>'
        )

    # --- media and health ------------------------------------------------------

    @app.get("/snapshots/{snapshot_id:path}")
    def snapshot(request: Request, snapshot_id: str) -> FileResponse:
        path = current(request).snapshots.path_for(snapshot_id)
        if path is None:
            raise HTTPException(status_code=404, detail="No such snapshot")
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/healthz", response_class=PlainTextResponse)
    def healthz() -> str:
        return "ok"

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics(request: Request) -> str:
        """A minimal readout, in Prometheus text format."""
        services = current(request)
        return "\n".join(
            [
                "# HELP reconvision_identities Enrolled identities.",
                "# TYPE reconvision_identities gauge",
                f"reconvision_identities {len(services.gallery.list_identities())}",
                "# HELP reconvision_gallery_entries Stored face descriptors.",
                "# TYPE reconvision_gallery_entries gauge",
                f"reconvision_gallery_entries {services.gallery.count_entries()}",
                "# HELP reconvision_events_stored Recognition events retained.",
                "# TYPE reconvision_events_stored gauge",
                f"reconvision_events_stored {services.events.count()}",
                "",
            ]
        )

    return app


def _cameras_in(events: Sequence[RecognitionEvent]) -> list[str]:
    return sorted({event.camera_name for event in events})


def _local_time(value: datetime) -> str:
    """Stored in UTC, read in the viewer's own clock."""
    return value.astimezone().strftime("%H:%M:%S")


def _scale_position(similarity: float) -> float:
    """Where a similarity sits on the 0-1 scale drawn in each row.

    Clamped rather than dropped: an event with no comparison at all still needs a
    defined position, and pinning it to the floor reads correctly as "no evidence".
    """
    return max(0.0, min(1.0, similarity)) * 100


__all__ = ["EventVerdict", "create_app"]
