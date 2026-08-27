"""Publishing events to MQTT, for Home Assistant and other automations.

Different in kind from a push notification: this is not for a human to read, it is
for the house to act on. Turning the hall light on when you come home, and not
arming anything when it is the cat, are the sort of rules this makes possible.
"""

from __future__ import annotations

import json

import paho.mqtt.client as mqtt
import structlog

from reconvision.domain.events import RecognitionEvent
from reconvision.domain.models import Frame

logger = structlog.get_logger(__name__)

DEFAULT_TOPIC_PREFIX = "reconvision"
_CONNECT_TIMEOUT_SECONDS = 5


class MqttNotifier:
    """Publishes one JSON message per event."""

    def __init__(
        self,
        host: str,
        port: int = 1883,
        username: str = "",
        password: str = "",
        topic_prefix: str = DEFAULT_TOPIC_PREFIX,
        client: mqtt.Client | None = None,
    ) -> None:
        if not host:
            message = "An MQTT host is required"
            raise ValueError(message)

        self._topic_prefix = topic_prefix.rstrip("/")
        self._client = client or mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        if username:
            self._client.username_pw_set(username, password)

        self._client.connect(host, port, keepalive=60)
        # A network loop in a background thread, so publishing never blocks the
        # pipeline waiting on a broker.
        self._client.loop_start()
        logger.info("mqtt_connected", host=host, port=port, prefix=self._topic_prefix)

    def notify(self, event: RecognitionEvent, snapshot: Frame | None = None) -> None:
        """Publish the event.

        The snapshot is deliberately not published: MQTT brokers are a poor fit for
        binary payloads of this size, and an automation needs to know that someone
        arrived, not what they looked like. The image stays in the snapshot store
        and reaches a human through ntfy.
        """
        topic = f"{self._topic_prefix}/{event.camera_name}/{event.verdict.value}"
        payload = {
            "event_id": event.event_id,
            "camera": event.camera_name,
            "verdict": event.verdict.value,
            "identity": event.identity_id,
            "animal": event.animal_label,
            "started_at": event.started_at.isoformat(),
            "duration_seconds": round(event.duration_seconds, 1),
            "confidence": round(event.confidence, 3),
            "observations": event.observations,
        }

        info = self._client.publish(
            topic,
            json.dumps(payload),
            qos=1,
            # Not retained: an event is something that happened at a moment, and a
            # retained one would tell every reconnecting automation that a stranger
            # is in the hall long after they left.
            retain=False,
        )
        info.wait_for_publish(timeout=_CONNECT_TIMEOUT_SECONDS)
        logger.debug("mqtt_published", topic=topic, event_id=event.event_id)

    def close(self) -> None:
        self._client.loop_stop()
        self._client.disconnect()


def home_assistant_discovery_payload(camera_name: str) -> tuple[str, dict[str, object]]:
    """A Home Assistant MQTT discovery message, so the camera appears by itself.

    Saves hand-editing YAML for every camera, which is exactly the kind of setup
    step that silently rots when a camera is renamed.
    """
    topic = f"homeassistant/sensor/reconvision_{camera_name}/config"
    payload: dict[str, object] = {
        "name": f"ReconVision {camera_name.replace('_', ' ')}",
        "unique_id": f"reconvision_{camera_name}",
        "state_topic": f"{DEFAULT_TOPIC_PREFIX}/{camera_name}/+",
        "value_template": "{{ value_json.verdict }}",
        "json_attributes_topic": f"{DEFAULT_TOPIC_PREFIX}/{camera_name}/+",
        "icon": "mdi:face-recognition",
        "device": {
            "identifiers": ["reconvision"],
            "name": "ReconVision",
            "manufacturer": "reconvision",
        },
    }
    return topic, payload
