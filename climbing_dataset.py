import cv2
import numpy as np
import sys
import os
import subprocess
import time
import pickle
from tqdm import tqdm

# Print everything
np.set_printoptions(threshold=np.inf)

# video level data structures
video_list = []

# frame level data structures
frame_list = []
frame_label_list = []

frame_list_val = []
frame_label_list_val = []



def save_label(filename):
    file1 = open(filename, 'r', errors='replace')
    labels = []
    lines = file1.readlines()

    for line in lines:
        if line.split("_")[0] == "frame":
            label = line.split(" ")[1].strip()
            if label.isnumeric():
                labels.append(int(label))

    return labels


def save_video(filename):

    frames = []

    # Creating a VideoCapture object to read the video
    # cap = cv2.VideoCapture('raw_videos/AndyLiu/Moonboard_Benchmarks_7b_They_Said_(7a__V6).mp4')
    cap = cv2.VideoCapture(filename)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buf_c = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
    buf = np.empty((frameCount, frameHeight, frameWidth), np.dtype('uint8'))

    fc = 0
    ret = True

    # Resolution reduction
    scale_percent = 30 # percent of original size
    width = int(buf.shape[2] * scale_percent / 100)
    height = int(buf.shape[1] * scale_percent / 100)
    dim = (width, height)

    # print(frameWidth, frameHeight, dim)

    # Loop until the end of the video
    i = 0
    while (fc < frameCount and ret and i < 30):
        ret, buf_c[fc] = cap.read()

        # convert to greyscale
        buf[fc] = cv2.cvtColor(buf_c[fc], cv2.COLOR_BGR2GRAY)
        # cv2.imshow("Final output", buf[fc])
        # cv2.waitKey(0)

        # reduce resolution
        resized = cv2.resize(buf[fc], dim, interpolation = cv2.INTER_AREA)

        frames.append(resized)
        fc += 1

    # release the video capture object
    cap.release()

    # Closes all the windows currently opened.
    cv2.destroyAllWindows()

    return frames

# View datapoint in video and check labels
def show_datapoint(filename):
    data = np.load("raw_videos/dataset/Moonboard_Benchmarks_Apple_Picking_(6c__V5).mp4_video.npy")
    label = np.load("raw_videos/dataset/Moonboard_Benchmarks_Apple_Picking_(6c__V5).mp4_label.npy")

    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (0,20)
    fontScale = .5
    fontColor = (0, 0, 0)
    lineType = 2

    for i in range(label.shape[0]):
        img = data[i,]
        test = 2
        if label[i] == 0:
            text = "Falling"
        elif label[i] == 1:
            text = "Climbing"
        else:
            text = "Other"

        cv2.putText(img, text,
            position,
            font,
            fontScale,
            fontColor,
            lineType)
        cv2.imshow("Final output", img)
        time.sleep(.05)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def dump_annotations(label_dict):
    # get completed tasks and dump annotations
    cmd = "/Users/testuser/Documents/cvat/utils/cli/cli.py --auth kwj2104:goodluck ls"

    child = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, encoding='utf8')
    output = child.communicate()[0].strip().split('\n')

    process_list = []
    for f in output:
        if f.split(",")[-1] == "completed":
            task_no = f.split(",")[0]
            dump_name = task_no + "_dump.txt"
            filename = f.split(",")[1].split(" ")[1]

            if not os.path.exists("raw_dump/" + dump_name):
                cmd_dump = "/Users/testuser/Documents/cvat/utils/cli/cli.py --auth kwj2104:goodluck dump --format \"ImageNet 1.0\" " + task_no + " " + "raw_dump/" + task_no + "_dump.txt"
                child = subprocess.Popen(cmd_dump, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, encoding='utf8')
                process_list.append(child)

            label_dict[filename] = dump_name

    for c in process_list:
        c.wait()


def generate_dataset(dump_anno=False, split_ratio=.8):
    label_dict = {}
    # load map from videos to tasks to labels
    print("Loading map for video annotation dump...")
    if dump_anno:
        dump_annotations(label_dict)
        filename_v = "raw_dump/label_dict.pkl"
        outfile_v = open(filename_v, 'wb')
        pickle.dump(label_dict, outfile_v)
        outfile_v.close()
    else:
        with open("raw_dump/label_dict.pkl", 'rb') as pickle_file:
            label_dict = pickle.load(pickle_file)

    print("Found " + str(len(label_dict)) + " video labels")

    print("Getting all labelled video files...")
    # get all video files, labelled ones only
    for base, dirs, files in os.walk('.'):
        for f in files:
            if f.endswith('.mp4') and f in label_dict:
                video_list.append((base, f))

    print("Found " + str(len(video_list)) + " corresponding videos")

    # generate entire dataset
    print("Generating dataset...")

    train_len = len(video_list) * split_ratio
    val_len = len(video_list) - train_len

    for i, video in tqdm(enumerate(video_list)):
        base, f = video
        frame_data = save_video(os.path.join(base, f))
        frame_label = save_label("raw_dump/" + label_dict[f])

        # Check dim
        # assert (frame_data[0].shape, (216, 121))

        # Split dataset into training and validation based on split_ratio
        if i < train_len:
            for j in range(len(frame_label)):
                frame_list.append(frame_data[j])
                frame_label_list.append(frame_label[j])
        else:
            for j in range(len(frame_label)):
                frame_list_val.append(frame_data[j])
                frame_label_list_val.append(frame_label[j])

        print("dim: ", frame_data[0].shape, f)

    print("Saving generated dataset...")
    filename_v = "climb_frame_dataset.pkl"
    outfile_v = open(filename_v, 'wb')
    pickle.dump(frame_list, outfile_v)
    outfile_v.close()

    filename_l = "climb_label_dataset.pkl"
    outfile_l = open(filename_l, 'wb')
    pickle.dump(frame_label_list, outfile_l)
    outfile_l.close()

    filename_v = "climb_frame_dataset_val.pkl"
    outfile_v = open(filename_v, 'wb')
    pickle.dump(frame_list_val, outfile_v)
    outfile_v.close()

    filename_l = "climb_label_dataset_val.pkl"
    outfile_l = open(filename_l, 'wb')
    pickle.dump(frame_label_list_val, outfile_l)
    outfile_l.close()

if __name__ == '__main__':
    generate_dataset(dump_anno=True)