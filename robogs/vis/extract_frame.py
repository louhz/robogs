import argparse
import os
import cv2

def extract_frames(video_path: str, image_path: str, frame_num: int) -> None:
    """
    Extract frames from the specified video and save them to the given directory.
    If frame_num >= total_frames, all frames will be extracted.
    Otherwise, frames are uniformly sampled based on frame_num.
    """
    # Create the image_path directory if it doesn't exist
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {total_frames}")

    # Decide whether to extract all frames or sample uniformly
    if frame_num >= total_frames:
        # Extract every frame
        print(f"Requested frames >= total frames. Extracting all {total_frames} frames.")
        frame_indices = range(total_frames)
    else:
        # Extract frames uniformly
        print(f"Extracting {frame_num} frames uniformly.")
        step = total_frames / frame_num
        frame_indices = [int(i * step) for i in range(frame_num)]

    extracted_count = 0
    # Loop through frames, saving only the chosen indices
    for idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        if idx in frame_indices:
            image_filename = os.path.join(image_path, f"frame_{idx:06d}.jpg")
            cv2.imwrite(image_filename, frame)
            extracted_count += 1

    cap.release()
    print(f"Done. Extracted {extracted_count} frames to {image_path}.")


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from a video uniformly."
    )
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Path to the input video."
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Directory to save extracted images."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        required=True,
        help="Number of frames to extract (if frame_num >= total_frames, extract all)."
    )

    args = parser.parse_args()

    # Call the extraction function
    extract_frames(args.video_path, args.image_path, args.frame_num)


if __name__ == "__main__":
    main()
