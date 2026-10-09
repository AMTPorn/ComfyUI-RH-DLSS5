# SPDX-License-Identifier: MIT
"""Disk-backed variants of the DLSS5 nodes for the AMT VideoHelperSuite fork.

VHS_DISK_MEDIA is a dict describing a working clip on disk ({path, fps, count,
width, height, pix_fmt, ephemeral, ...}; lossless ffv1/gbrp .mkv under
/root/autodl-tmp/vhs_disk). These nodes decode it frame by frame, run DLSS5
and encode the result into a new working clip, so no IMAGE batch is ever held
in RAM. Clip naming, colour handling and reference counting (deleting spent
intermediate clips) are delegated to the VHS disk module itself so the clips
behave exactly like the ones produced by the VHS disk nodes.
"""

from __future__ import annotations

import subprocess
import sys
import threading

import numpy as np

from .dlss5nr.common import DLSS5Error

DISK_TYPE = "VHS_DISK_MEDIA"
DEFAULT_BASE_DIR = "/root/autodl-tmp/vhs_disk"


def _vhs():
    """The loaded VHS disk module (ComfyUI registers custom nodes under
    path-derived module names, so look it up by suffix instead of importing)."""
    for name, module in list(sys.modules.items()):
        if name.endswith("videohelpersuite.disk_nodes") and \
                getattr(module, "DISK_TYPE", None) == DISK_TYPE:
            return module
    raise DLSS5Error(
        "Disk nodes need ComfyUI-VideoHelperSuite with disk nodes "
        "(github.com/SamTyurenkov/ComfyUI-VideoHelperSuite) loaded.")


class _DiskReader:
    """Sequential uint8 RGB reader over a VHS_DISK_MEDIA clip.

    Same interface as decoder.ChunkedVideoReader (width/height/count/fps +
    iter_frames), so it plugs into the existing streaming paths. Decoding goes
    through VHS's colour filters: working clips are full-range RGB, delivery
    files may be TV-range BT.709 YUV.
    """

    def __init__(self, vhs, media: dict):
        self.vhs = vhs
        self.media = media
        self.path = media["path"]
        self.width = int(media["width"])
        self.height = int(media["height"])
        self.count = int(media["count"])
        self.fps = float(media["fps"] or 24.0)
        if self.width <= 0 or self.height <= 0 or self.count <= 0:
            raise DLSS5Error(f"Disk media has invalid size/count: {self.path}")

    def iter_frames(self):
        vhs = self.vhs
        frame_bytes = self.width * self.height * 3
        args = [
            vhs._ffmpeg(), "-v", "error", "-i", self.path, "-vsync", "0", "-an",
            *vhs.decode_to_rgb_vf(self.media),
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ]
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                bufsize=frame_bytes)
        errors = []
        drain = threading.Thread(target=lambda: errors.append(proc.stderr.read()), daemon=True)
        drain.start()
        index = 0
        try:
            while index < self.count:
                raw = proc.stdout.read(frame_bytes)
                if len(raw) < frame_bytes:
                    break
                frame = np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 3)
                yield index, frame
                index += 1
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
            drain.join(timeout=5)
        if index < self.count:
            err = b"".join(e for e in errors if e).decode("utf-8", errors="replace").strip()
            raise DLSS5Error(
                f"disk decode ended at frame {index}/{self.count} ({self.path})"
                + (f":\n{err}" if err else ""))


def _write_clip(vhs, base_dir, unique_id, prefix, width, height, fps, produce):
    """Run `produce(sink)` into a new VHS working clip; return the raw media dict."""
    out_path = vhs.new_clip_path(base_dir, unique_id, prefix)
    writer = vhs._FfmpegFrameWriter(out_path, width, height, float(fps), None)
    try:
        result = produce(lambda rgb: writer.write_bytes(np.ascontiguousarray(rgb).tobytes()))
        media = writer.close()
    except BaseException:
        try:
            writer.proc.kill()
            writer.proc.wait()
        except Exception:
            pass
        raise
    return media, result


def _optional_inputs(node_cls, drop):
    inputs = dict(node_cls.INPUT_TYPES()["optional"])
    for key in drop:
        inputs.pop(key, None)
    return inputs


class RH_DLSS5EnhanceDisk:
    """DLSS 5 Neural Rendering over a VHS disk clip: VHS_DISK_MEDIA in and out."""

    @classmethod
    def INPUT_TYPES(cls):  # noqa: N802 (ComfyUI convention)
        from .nodes import RH_DLSS5Enhance

        optional = _optional_inputs(RH_DLSS5Enhance, ("image", "video", "keep_audio"))
        style, style_opts = optional.pop("style")
        # style=off is a passthrough; use ComfyUI bypass (Ctrl+B) on this node instead.
        style = [s for s in style if not s.startswith("off")]
        return {
            "required": {
                "disk": (DISK_TYPE,),
                "style": (style, {**style_opts, "default": style[0]}),
                **optional,
                "base_dir": ("STRING", {"default": DEFAULT_BASE_DIR, "tooltip": "Where the output working clip is written (VHS disk root rules apply)."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID", "prompt": "PROMPT"},
        }

    RETURN_TYPES = (DISK_TYPE, "INT", "STRING")
    RETURN_NAMES = ("disk", "count", "status")
    FUNCTION = "enhance"
    CATEGORY = "video/disk"
    DESCRIPTION = (
        "RH DLSS5 Enhance for VideoHelperSuite disk clips: frames are decoded from the "
        "working clip one at a time and the result is written to a new lossless working "
        "clip, so long or high-resolution videos never sit in RAM as an IMAGE batch."
    )

    def enhance(self, disk, base_dir=DEFAULT_BASE_DIR, unique_id=None, prompt=None, **kwargs):
        from . import nodes

        vhs = _vhs()
        reader = _DiskReader(vhs, vhs.require_media(disk))
        backend_name = nodes._pick_backend(kwargs.get("backend", "auto"))
        params, auto_note = nodes._resolve_auto_scale(
            nodes._build_params(kwargs), reader.width, reader.height)
        out_w, out_h = nodes.common.target_size(reader.width, reader.height, params["scale"])
        media, (iw, ih, ow, oh, order, engine) = _write_clip(
            vhs, base_dir, unique_id, "dlss5", out_w, out_h, reader.fps,
            lambda sink: nodes._stream_frames(reader, params, backend_name, sink))
        media = vhs.as_working_media(media, prompt, unique_id)
        vhs.release_media(disk)
        status = (f"DISK {reader.count} frames @{reader.fps:g}fps {iw}x{ih}->{ow}x{oh} "
                  f"via {backend_name} (channel {order}, motion {engine}"
                  f"{', hdr' if params['hdr'] else ''}) -> {media['path']}"
                  + (f"; {auto_note}" if auto_note else ""))
        return (media, media["count"], status)


class RH_DLSS5FrameInterpolationDisk:
    """NVIDIA DLSS frame generation over a VHS disk clip: VHS_DISK_MEDIA in and out."""

    @classmethod
    def INPUT_TYPES(cls):  # noqa: N802 (ComfyUI convention)
        from .nodes import RH_DLSS5FrameInterpolation

        optional = _optional_inputs(
            RH_DLSS5FrameInterpolation, ("image", "video", "audio", "keep_audio"))
        choices, fps_opts = optional.pop("output_fps")
        # 1x is a passthrough; use ComfyUI bypass (Ctrl+B) on this node instead.
        choices = [c for c in choices if c != "1x"]
        return {
            "required": {
                "disk": (DISK_TYPE,),
                "output_fps": (choices, {**fps_opts, "default": "2x"}),
                **optional,
                "base_dir": ("STRING", {"default": DEFAULT_BASE_DIR, "tooltip": "Where the output working clip is written (VHS disk root rules apply)."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID", "prompt": "PROMPT"},
        }

    RETURN_TYPES = (DISK_TYPE, "INT", "STRING")
    RETURN_NAMES = ("disk", "count", "status")
    FUNCTION = "interpolate"
    CATEGORY = "video/disk"
    DESCRIPTION = (
        "RH DLSS5 frame generation for VideoHelperSuite disk clips: the source clip is "
        "decoded sequentially and generated frames are streamed into a new lossless "
        "working clip at the output frame rate. Working clips carry no audio."
    )

    def interpolate(self, disk, output_fps="2x", base_dir=DEFAULT_BASE_DIR,
                    unique_id=None, prompt=None, **kwargs):
        from comfy.utils import ProgressBar

        from .dlss5nr.decoder import SequentialSource
        from .dlss5nr.fg import MULTIPLIERS, as_fps, interpolate_stream_from_source

        vhs = _vhs()
        reader = _DiskReader(vhs, vhs.require_media(disk))
        choice = str(output_fps or "2x")
        if choice in MULTIPLIERS:
            multiplier, target_fps = MULTIPLIERS[choice], None
        else:
            multiplier, target_fps = 2, as_fps(choice)
        src_fps = as_fps(reader.fps)
        if target_fps is not None and target_fps <= src_fps:
            raise DLSS5Error(
                f"target output fps ({float(target_fps):g}) must exceed the source "
                f"rate ({float(src_fps):g}); interpolation only increases frame rate.")
        out_fps = target_fps if target_fps is not None else src_fps * multiplier
        bar = ProgressBar(100)
        media, (out_fps, note) = _write_clip(
            vhs, base_dir, unique_id, "dlssg", reader.width, reader.height, out_fps,
            lambda sink: interpolate_stream_from_source(
                SequentialSource(reader), reader.count, reader.width, reader.height,
                src_fps, multiplier, str(kwargs.get("motion", "auto")),
                float(kwargs.get("scene_change_threshold", 0.24) or 0.24),
                str(kwargs.get("runtime_dir", "") or ""), str(kwargs.get("wine_prefix", "") or ""),
                progress_callback=lambda done, total: bar.update_absolute(100 * done // max(1, total)),
                frame_sink=sink, target_fps=target_fps))
        media = vhs.as_working_media(media, prompt, unique_id)
        vhs.release_media(disk)
        status = (f"DISK {reader.count}->{media['count']} frames "
                  f"@{float(src_fps):g}->{float(out_fps):g}fps -> {media['path']}; {note}")
        return (media, media["count"], status)
