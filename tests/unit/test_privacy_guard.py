"""The privacy guard is the last line of defence for a public repository.

A leak here is irreversible once pushed, so the guard is tested against the two
mistakes it exists to catch, and against the look-alikes it must not reject.
"""

from pathlib import Path

import pytest

from reconvision.tooling.privacy_guard import main

CREDENTIALS = "admin:hunter2"


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway working tree, since the guard resolves paths relative to the cwd."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "docs").mkdir()
    return tmp_path


def test_rejects_a_captured_frame(repo: Path) -> None:
    (repo / "face.jpg").write_bytes(b"\xff\xd8\xff")

    assert main(["face.jpg"]) == 1


def test_rejects_the_event_database(repo: Path) -> None:
    (repo / "reconvision.db").write_bytes(b"SQLite format 3\x00")

    assert main(["reconvision.db"]) == 1


def test_rejects_face_embeddings(repo: Path) -> None:
    (repo / "gallery.npy").write_bytes(b"\x93NUMPY")

    assert main(["gallery.npy"]) == 1


def test_rejects_a_stream_url_carrying_credentials(repo: Path) -> None:
    # Assembled rather than written literally: a literal would make this very file
    # trip the guard it is testing, which is exactly the behaviour under test.
    (repo / "cameras.yaml").write_text(f"source: rtsp://{CREDENTIALS}@192.168.1.50:554/s1\n")

    assert main(["cameras.yaml"]) == 1


def test_allows_documentation_assets(repo: Path) -> None:
    (repo / "docs" / "screenshot.png").write_bytes(b"\x89PNG")

    assert main(["docs/screenshot.png"]) == 0


def test_allows_a_stream_url_referencing_an_environment_variable(repo: Path) -> None:
    (repo / "cameras.yaml").write_text("source: ${LIVING_ROOM_STREAM_URL}\n")

    assert main(["cameras.yaml"]) == 0


def test_allows_a_credentialless_stream_url(repo: Path) -> None:
    (repo / "cameras.yaml").write_text("source: rtsp://192.168.1.50:554/stream1\n")

    assert main(["cameras.yaml"]) == 0


def test_allows_the_example_file_documenting_the_secret_shape(repo: Path) -> None:
    (repo / "cameras.example.yaml").write_text(f"source: rtsp://{CREDENTIALS}@192.168.1.50/s1\n")

    assert main(["cameras.example.yaml"]) == 0


def test_ignores_paths_that_do_not_exist(repo: Path) -> None:
    """Pre-commit passes deleted files too; a removal is not a leak."""
    assert main(["deleted.jpg"]) == 0


def test_survives_a_binary_file_it_cannot_decode(repo: Path) -> None:
    (repo / "notes.txt").write_bytes(b"\xff\xfe\x00\x80 not utf-8")

    assert main(["notes.txt"]) == 0


def test_allows_any_file_named_as_an_example(repo: Path) -> None:
    """Example files document the shape of a secret, so they cannot be forbidden
    from containing one. Matched by convention rather than a list of names, which
    would silently exclude the next template someone adds."""
    (repo / "go2rtc.example.yaml").write_text(f"source: rtsp://{CREDENTIALS}@192.168.1.50/s1\n")

    assert main(["go2rtc.example.yaml"]) == 0


def test_a_file_merely_mentioning_example_is_not_exempt(repo: Path) -> None:
    """The exemption is for `*.example.<ext>`, not for anything with the word in it."""
    (repo / "example-cameras.yaml").write_text(f"source: rtsp://{CREDENTIALS}@192.168.1.50/s1\n")

    assert main(["example-cameras.yaml"]) == 1
