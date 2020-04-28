# imports
import os
import sys
import argparse
import torch
import cv2
from PIL import Image
torch.manual_seed(1)


# local imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(project_root)
from data_handling.utils import list_files_rec, list_dir


########################################################################################################################
# functions
########################################################################################################################
def vid_to_images(vid_path, img_size, skip_frames=5):
    vidcap = cv2.VideoCapture(vid_path)
    images = []
    cnt = 0
    has_frame = True
    while has_frame:
        has_frame, frame = vidcap.read()
        if has_frame:
            if cnt % skip_frames == 0 and has_frame:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                curr_img_size = img.shape[1]
                if curr_img_size >= img_size:
                    img = Image.fromarray(img)
                    img = img.resize((img_size, img_size))
                    images.append(img)
                else:
                    print("Warning: video {} is smaller then {} pixels".format(vid_path, img_size))
        cnt += 1
    vidcap.release()
    return images


def id_vids_to_imgs(id_src_root_dir, id_dst_root_dir, img_size, skip_frames=5):
    """"""
    # create dst dir
    if not os.path.isdir(id_dst_root_dir):
        os.makedirs(id_dst_root_dir)

    # list mp4 files recursively
    src_vid_path_list = [vid_path for vid_path in list_files_rec(id_src_root_dir, suffix=('.mp4',))]

    # for each file: load, take frames, resize
    id_images = []
    for vid_path in src_vid_path_list:
        id_images.extend(vid_to_images(vid_path=vid_path, img_size=img_size, skip_frames=skip_frames))

    # save images
    for i, img in enumerate(id_images):
        img_path = os.path.join(id_dst_root_dir, "{:08}.jpg".format(i))
        img.save(img_path)


def vids_to_id_imgs(id_src_root_dir, id_dst_root_dir, img_size, skip_frames=5):
    """"""
    # list mp4 files recursively
    src_vid_path_list = [vid_path for vid_path in list_files_rec(id_src_root_dir, suffix=('.mp4',))]

    # for each file: load, take frames, resize
    for vid_idx, vid_path in enumerate(src_vid_path_list):
        # print("Processing vid: {}".format(vid_path))
        vid_dst_dir = os.path.join(id_dst_root_dir,'{:04}'.format(vid_idx))
        if not os.path.isdir(vid_dst_dir):
            os.makedirs(vid_dst_dir)
        vid_images = vid_to_images(vid_path=vid_path, img_size=img_size, skip_frames=skip_frames)

        # save images
        for img_idx, img in enumerate(vid_images):
            img_path = os.path.join(vid_dst_dir, "{:08}.jpg".format(img_idx))
            img.save(img_path)


def id_largest_vid_to_imgs(id_src_root_dir, id_dst_root_dir, img_size, skip_frames=5):
    """"""
    # create dst dir
    if not os.path.isdir(id_dst_root_dir):
        os.makedirs(id_dst_root_dir)

    # list mp4 files recursively
    src_vid_path_list = [vid_path for vid_path in list_files_rec(id_src_root_dir, suffix=('.mp4',))]

    # for each file: load, take frames, resize
    id_vid_images = []
    for vid_path in src_vid_path_list:
        id_vid_images.append(vid_to_images(vid_path=vid_path, img_size=img_size, skip_frames=skip_frames))

    id_images = max(id_vid_images, key=lambda x: len(x))
    # save images
    for i, img in enumerate(id_images):
        img_path = os.path.join(id_dst_root_dir, "{:08}.jpg".format(i))
        img.save(img_path)


def create_dataset(src_vid_ds_root, dst_img_ds_root, img_size, skip_frames=5):
    id_dir_list = list_dir(src_vid_ds_root)
    for id_dir in id_dir_list:
        src_id_dir = os.path.join(src_vid_ds_root, id_dir)
        dst_id_dir = os.path.join(dst_img_ds_root, id_dir)
        print("Processing dir: {}".format(src_id_dir))
        vids_to_id_imgs(id_src_root_dir=src_id_dir, id_dst_root_dir=dst_id_dir, img_size=img_size, skip_frames=skip_frames)


def main(args):
    create_dataset(
        src_vid_ds_root=args.src_vid_ds_root,
        dst_img_ds_root=args.dst_img_ds_root,
        img_size=args.img_size,
        skip_frames=args.skip_frames
    )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_vid_ds_root',
                        required=True,
                        help='path to voxceleb2 video directory at .../test/mp4/ or /dev/mp4')
    parser.add_argument('--dst_img_ds_root',
                        required=True,
                        help='location of new dataset')
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--skip_frames', type=int, default=5)
    return parser.parse_args()


########################################################################################################################
# App
########################################################################################################################
if __name__ == '__main__':
    main(get_args())




