import cv2
import os
from natsort import natsorted

def images_to_video(folders, output_video_name='output.mp4', fps=20):
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    
    # Gather all image paths from the specified folders in sequence
    all_image_paths = []
    for folder_path in folders:
        images_in_folder = [os.path.join(folder_path, f) 
                            for f in os.listdir(folder_path) 
                            if f.lower().endswith(valid_extensions)]
        # Sort images in natural order (1, 2, 10 instead of 1, 10, 2)
        images_in_folder = natsorted(images_in_folder)
        all_image_paths.extend(images_in_folder)
    
    if not all_image_paths:
        print("No valid images found in the folders.")
        return
    
    # Read the first image to get dimensions
    first_image_path = all_image_paths[0]
    frame = cv2.imread(first_image_path)
    if frame is None:
        print(f"Error: Unable to read the first image ({first_image_path}).")
        return
    height, width, channels = frame.shape
    
    # Define the codec for output video (mp4v typically works well for .mp4 files)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Create VideoWriter object
    # Here we assume that all images have the same dimensions
    out = cv2.VideoWriter(output_video_name, fourcc, fps, (width, height))
    
    # Write each image to the video
    for img_path in all_image_paths:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Warning: Unable to read {img_path}. Skipping.")
            continue
        out.write(frame)
    
    # Release the video writer
    out.release()
    print(f"Video saved to {output_video_name}")

if __name__ == "__main__":
    # Example usage:
    folder_list = [
        "/home/haozhe/Dropbox/rendering/asset/video/view2",
        # "/home/haozhe/Dropbox/rendering/asset/video/final",

    ]
    images_to_video(folder_list, output_video_name='demo4.mp4', fps=20)