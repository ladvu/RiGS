import argparse
import imageio.v3 as iio
import glob
import os

def image_sequence_to_video(
    image_seq_path:str,
    video_output_path:str,
    fps:int=12
):
    image_files = sorted(glob.glob(os.path.join(image_seq_path, "*")))
    image_files = sorted([f for f in image_files if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    frames = [iio.imread(f) for f in image_files]
    iio.imwrite(video_output_path, frames, fps=fps, macro_block_size=1)

def video_to_image_sequence(
    video_path:str,
    image_seq_output:str
):
    os.makedirs(image_seq_output, exist_ok=True)
    frames = iio.imread(video_path)
    for i, frame in enumerate(frames):
        iio.imwrite(os.path.join(image_seq_output, f"{i:05d}.png"), frame)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()

    img_dir = os.path.join(args.data_root, "images")
    video_dir = os.path.join(args.data_root, "videos")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    image_seqs = os.listdir( img_dir)
    videos = [p.split(".")[0] for p in os.listdir(video_dir)]

    scenes = list(set(image_seqs).union(set(videos)))

    for scene in scenes:
        img_seq_dir = os.path.join(img_dir, scene)
        video_matches = glob.glob(os.path.join(video_dir, f"{scene}.*"))
        video_path = video_matches[0] if video_matches else os.path.join(video_dir, f"{scene}.mp4")
        if os.path.exists(img_seq_dir) and video_matches:
            continue
        elif os.path.exists(img_seq_dir):
            image_sequence_to_video(img_seq_dir, video_path, fps=args.fps)
        else:
            video_to_image_sequence(video_path, img_seq_dir)


    

