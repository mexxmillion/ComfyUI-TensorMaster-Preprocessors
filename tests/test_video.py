import subprocess

import pytest

from video import prepare_video


def ffmpeg_executable():
    import shutil

    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    return pytest.importorskip("imageio_ffmpeg").get_ffmpeg_exe()


def test_prepare_video_is_bounded_cached_and_preserves_source(tmp_path):
    source = tmp_path / "h3_mobile" / "source.mp4"
    source.parent.mkdir()
    subprocess.run(
        [
            ffmpeg_executable(), "-v", "error", "-f", "lavfi", "-i",
            "testsrc2=size=96x160:rate=30:duration=2", "-f", "lavfi", "-i",
            "sine=frequency=440:duration=2", "-c:v", "libx264", "-c:a", "aac",
            "-shortest", "-y", str(source),
        ],
        check=True,
        capture_output=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    original = source.read_bytes()
    relative = prepare_video(
        tmp_path, "h3_mobile/source.mp4", 64, 96, 22, trim_json="[0.5, 1.5]"
    )
    result = tmp_path / relative
    assert result.is_file() and source.read_bytes() == original
    assert prepare_video(
        tmp_path, "h3_mobile/source.mp4", 64, 96, 22, trim_json="[0.5, 1.5]"
    ) == relative


@pytest.mark.parametrize(
    "source,width,height,frames,crop,trim",
    [
        ("../outside.mp4", 64, 96, 22, "null", "null"),
        ("h3_mobile/source.mp4", 64, 96, 22, "[0,0,2,1]", "null"),
        ("h3_mobile/source.mp4", 64, 96, 22, "null", "[2,1]"),
        ("h3_mobile/source.mp4", 16, 96, 22, "null", "null"),
    ],
)
def test_prepare_video_rejects_unmanaged_or_invalid_inputs(
    tmp_path, source, width, height, frames, crop, trim
):
    managed = tmp_path / "h3_mobile" / "source.mp4"
    managed.parent.mkdir()
    managed.touch()
    with pytest.raises((ValueError, RuntimeError)):
        prepare_video(tmp_path, source, width, height, frames, crop, trim)
