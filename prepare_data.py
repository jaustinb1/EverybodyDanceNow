import os
import subprocess
import argparse

OPENPOSE_LOCATION = "/home/jonathan/installed_libraries/openpose/build/examples/openpose/openpose.bin"
OPENPOSE_BASE = "/home/jonathan/installed_libraries/openpose"

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, required=True)
parser.add_argument('--rate', type=int, default=120)
parser.add_argument('--destination', type=str, required=True)
parser.add_argument('--source_keypoints', type=str)
parser.add_argument('--train', action='store_true')

def split_video(source, rate, destination):
    print("Splitting the video into frames")
    original_frame_dir = os.path.join(destination, 'original_frames')

    if not os.path.exists(original_frame_dir):
        os.makedirs(original_frame_dir)

    # first convert the video to frames
    proc = subprocess.Popen([
        'ffmpeg', '-i', '{}'.format(source),
        '-r', '{}'.format(rate),
        #'-f',
        #'frame-%7d.jpg',
        '{}/frame-%7d.jpg'.format(original_frame_dir)
    ])
    proc.wait()

    print('Done splitting the video')
    return original_frame_dir

def run_openpose(source, destination, is_train):
    print("Running openpose pose estimation")

    open_pose_result_location = os.path.join(
        destination, "jason_{}".format('train' if is_train else 'eval')
    )

    source = os.path.abspath(source)
    open_pose_result_location = os.path.abspath(open_pose_result_location)

    cwd = os.getcwd()
    os.chdir(OPENPOSE_BASE)
    proc = subprocess.Popen([
        "./build/examples/openpose/openpose.bin", "--video={}".format(source),
        '--face', '--hand',
        '--write_json={}'.format(open_pose_result_location)
    ])
    proc.wait()
    os.chdir(cwd)

    print("Finished running openpose")
    print("Renaming open pose results")

    for keypoint_file in list(os.listdir(open_pose_result_location)):
        try:
            new_keypoint_file = "frame-{0:07d}_keypoints.json".format(int(keypoint_file.split("_")[-2]))
        except:
            break
        os.rename(
            os.path.join(open_pose_result_location, keypoint_file),
            os.path.join(open_pose_result_location, new_keypoint_file)
        )

    return open_pose_result_location


def prep_train_data(original_frames, openpose_res_dir, destination):

    #python graph_train.py --keypoints_dir=/home/jonathan/Desktop/everybodydance_data/train/jason_train --frames_dir=/home/jonathan/Desktop/everybodydance_data/train/original_frames/ --save_dir=/home/jonathan/Desktop/everybodydance_data/train/ --spread 4000 25631 1

    print("Graphing train data")
    end = len(list(os.listdir(original_frames)))

    proc = subprocess.Popen([
        "python", "data_prep/graph_train.py",
        "--keypoints_dir={}".format(openpose_res_dir),
        "--frames_dir={}".format(original_frames),
        "--save_dir={}".format(destination),
        "--spread", "0", "{}".format(end), "1"
    ])
    proc.wait()
    print("Finished graphing train data")

def prep_test_data(og_frame_dir, source_keypoints, target_keypoints, destination):

    print("Graphing test data")
    source_end = len(list(os.listdir(source_keypoints)))
    target_end = len(list(os.listdir(target_keypoints)))
    proc = subprocess.Popen([
        "python", "data_prep/graph_posenorm.py",
        "--source_keypoints={}".format(target_keypoints),
        "--target_keypoints={}".format(source_keypoints),
        "--source_spread", "0", "{}".format(target_end),
        "--target_spread", "0", "{}".format(source_end),
        "--results={}".format(destination),
        "--calculate_scale_translation"
    ])
    proc.wait()
    print("Finished graphing test data")

if __name__ == '__main__':
    args = parser.parse_args()

    og_frame_dir = split_video(args.source, args.rate, args.destination)
    pose_dir = run_openpose(args.source, args.destination, args.train)

    if args.train:
        prep_train_data(og_frame_dir, pose_dir, args.destination)
    else:
        prep_test_data(og_frame_dir, args.source_keypoints, pose_dir, args.destination)
