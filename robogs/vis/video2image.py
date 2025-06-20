import math
import shutil
import subprocess
import argparse
from pathlib import Path
from typing import Tuple

def run_command(command: str):
    subprocess.run(command, shell=True, check=True)


def get_num_frames_in_video(video_path: Path) -> int:
    cmd_duration = (
        f"ffprobe -v error -select_streams v:0 "
        f"-show_entries format=duration "
        f"-of default=noprint_wrappers=1:nokey=1 '{video_path}'"
    )
    cmd_fps = (
        f"ffprobe -v error -select_streams v:0 "
        f"-show_entries stream=r_frame_rate "
        f"-of default=noprint_wrappers=1:nokey=1 '{video_path}'"
    )

    # Get duration
    duration_result = subprocess.run(cmd_duration, shell=True, capture_output=True, text=True)
    duration_str = duration_result.stdout.strip()
    duration = float(duration_str) if duration_str else 0.0

    # Get frame rate
    fps_result = subprocess.run(cmd_fps, shell=True, capture_output=True, text=True)
    fps_str = fps_result.stdout.strip().split('\n')[0]
    if '/' in fps_str:
        num, den = fps_str.split('/')
        fps = float(num) / float(den)
    else:
        fps = float(fps_str) if fps_str else 0.0

    # Calculate frames
    num_frames = int(duration * fps) if duration and fps else 0
    return num_frames

def convert_video_to_images(
    video_path: Path,
    image_dir: Path,
    num_frames_target: int,
    num_downscales: int,
    crop_factor: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    image_prefix: str = "frame_",
    keep_image_dir: bool = False,
):
    if not keep_image_dir:
        for i in range(num_downscales + 1):
            dir_to_remove = image_dir if i == 0 else Path(f"{image_dir}_{2**i}")
            shutil.rmtree(dir_to_remove, ignore_errors=True)
    image_dir.mkdir(exist_ok=True, parents=True)

    num_frames = get_num_frames_in_video(video_path)
    spacing = max(1, num_frames // num_frames_target)

    ffmpeg_cmd = f'ffmpeg -i "{video_path}"'

    crop_cmd = ""
    if crop_factor != (0.0, 0.0, 0.0, 0.0):
        height = 1 - crop_factor[0] - crop_factor[1]
        width = 1 - crop_factor[2] - crop_factor[3]
        start_x = crop_factor[2]
        start_y = crop_factor[0]
        crop_cmd = f"crop=w=iw*{width}:h=ih*{height}:x=iw*{start_x}:y=ih*{start_y},"

    downscale_chains = [f"[t{i}]scale=iw/{2**i}:ih/{2**i}[out{i}]" for i in range(num_downscales + 1)]
    downscale_dirs = [Path(str(image_dir) + (f"_{2**i}" if i > 0 else "")) for i in range(num_downscales + 1)]
    downscale_paths = [downscale_dirs[i] / f"{image_prefix}%05d.png" for i in range(num_downscales + 1)]

    for dir in downscale_dirs:
        dir.mkdir(parents=True, exist_ok=True)

    downscale_chain = (
        f"split={num_downscales + 1}"
        + "".join([f"[t{i}]" for i in range(num_downscales + 1)])
        + ";"
        + ";".join(downscale_chains)
    )

    ffmpeg_cmd += " -vsync vfr"
    select_cmd = f"thumbnail={spacing},setpts=N/TB," if spacing > 1 else ""

    downscale_cmd = f' -filter_complex "{select_cmd}{crop_cmd}{downscale_chain}"' + "".join(
        [f' -map "[out{i}]" "{downscale_paths[i]}"' for i in range(num_downscales + 1)]
    )

    ffmpeg_cmd += downscale_cmd

    run_command(ffmpeg_cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert video to images.")
    parser.add_argument("-v", "--video", required=True, type=str, help="Path to the video file.")
    parser.add_argument("-o", "--output", required=True, type=str, help="Path to the output image directory.")
    parser.add_argument("--num_frames", default=100, type=int, help="Number of frames to target.")
    parser.add_argument("--num_downscales", default=2, type=int, help="Number of downscales.")
    parser.add_argument("--crop_factor", nargs=4, type=float, default=[0.0, 0.0, 0.0, 0.0], help="Crop factor (top, bottom, left, right).")
    parser.add_argument("--image_prefix", default="frame_", type=str, help="Prefix for image filenames.")
    parser.add_argument("--keep_image_dir", action='store_true', help="Keep existing image directory.")

    args = parser.parse_args()

    convert_video_to_images(
        video_path=Path(args.video),
        image_dir=Path(args.output),
        num_frames_target=args.num_frames,
        num_downscales=args.num_downscales,
        crop_factor=tuple(args.crop_factor),
        image_prefix=args.image_prefix,
        keep_image_dir=args.keep_image_dir
    )
