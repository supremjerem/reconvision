"""Configuration, resolved from the environment and a camera manifest.

Twelve-factor: everything environment-specific arrives through env vars, and
nothing secret is ever written to a versioned file. Camera credentials in
particular live only in `.env`; `cameras.yaml` refers to them by name.
"""

from __future__ import annotations

import os
import re
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Self

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: `${VARIABLE}` references inside the camera manifest.
_ENV_REFERENCE = re.compile(r"\$\{([A-Z0-9_]+)\}")


class TelemetryExporter(StrEnum):
    """Where traces and metrics go."""

    #: Human-readable, for local development.
    CONSOLE = "console"
    #: OTLP over HTTP, for a Grafana/Tempo/Prometheus stack.
    OTLP = "otlp"
    NONE = "none"


class CameraConfig(BaseModel):
    """One video source to watch."""

    name: str = Field(min_length=1, pattern=r"^[a-z0-9_]+$")
    source: str = Field(min_length=1)
    #: Analyse one frame in N. Motion gating already skips still scenes; this caps
    #: the cost while something is genuinely moving.
    sample_every_n_frames: int = Field(default=3, ge=1, le=60)
    enabled: bool = True


class Settings(BaseSettings):
    """Everything the application needs to run."""

    model_config = SettingsConfigDict(
        env_prefix="RECONVISION_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = Path("./data")
    snapshot_retention_days: int = Field(default=30, ge=1)

    # --- Recognition ---------------------------------------------------------
    #: Calibrate with `reconvision eval` rather than guessing. The default is a
    #: reasonable starting point for buffalo_l, not a tuned value.
    match_threshold: Annotated[float, Field(ge=-1.0, le=1.0)] = 0.42
    min_match_margin: Annotated[float, Field(ge=0.0)] = 0.05
    face_model_pack: str = "buffalo_l"
    min_face_pixels: int = Field(default=80, ge=16)

    # --- Notifications -------------------------------------------------------
    ntfy_url: str = "https://ntfy.sh"
    ntfy_topic: str = ""
    mqtt_host: str = ""
    mqtt_port: int = 1883
    mqtt_username: str = ""
    mqtt_password: str = ""
    webhook_url: str = ""

    # --- Observability -------------------------------------------------------
    telemetry_exporter: TelemetryExporter = TelemetryExporter.CONSOLE
    otlp_endpoint: str = "http://localhost:4318"
    log_level: str = "INFO"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def gallery_dir(self) -> Path:
        return self.data_dir / "gallery"

    @property
    def snapshots_dir(self) -> Path:
        return self.data_dir / "snapshots"

    @property
    def database_path(self) -> Path:
        return self.data_dir / "reconvision.db"

    @property
    def enabled_notifiers(self) -> tuple[str, ...]:
        """Which delivery channels are configured.

        Absence of configuration is how a channel is turned off; there is no
        separate enable flag to fall out of sync with it.
        """
        configured = {
            "ntfy": bool(self.ntfy_topic),
            "mqtt": bool(self.mqtt_host),
            "webhook": bool(self.webhook_url),
        }
        return tuple(name for name, is_configured in configured.items() if is_configured)


class CameraManifest(BaseModel):
    """The parsed contents of `cameras.yaml`."""

    cameras: list[CameraConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def reject_duplicate_names(self) -> Self:
        names = [camera.name for camera in self.cameras]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            message = f"Camera names must be unique, repeated: {sorted(duplicates)}"
            raise ValueError(message)
        return self

    @property
    def enabled(self) -> list[CameraConfig]:
        return [camera for camera in self.cameras if camera.enabled]


class MissingCameraCredentialError(RuntimeError):
    """A camera refers to an environment variable that is not set."""

    def __init__(self, variable: str, camera: str) -> None:
        super().__init__(
            f"Camera {camera!r} refers to ${{{variable}}}, which is not set. "
            f"Add it to your .env file."
        )
        self.variable = variable


def _expand(value: str, camera_name: str, environ: dict[str, str]) -> str:
    """Substitute `${VAR}` references, failing loudly on a missing one.

    Failing loudly matters: silently leaving the placeholder in place would turn
    a missing password into an unreadable connection error hours later.
    """

    def replace(match: re.Match[str]) -> str:
        variable = match.group(1)
        if variable not in environ:
            raise MissingCameraCredentialError(variable, camera_name)
        return environ[variable]

    return _ENV_REFERENCE.sub(replace, value)


def load_camera_manifest(path: Path, environ: dict[str, str] | None = None) -> CameraManifest:
    """Read `cameras.yaml` and resolve credential references from the environment."""
    resolved_environ = dict(os.environ) if environ is None else environ

    if not path.exists():
        return CameraManifest()

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    manifest = CameraManifest.model_validate(raw)

    return CameraManifest(
        cameras=[
            camera.model_copy(
                update={"source": _expand(camera.source, camera.name, resolved_environ)}
            )
            for camera in manifest.cameras
        ]
    )
