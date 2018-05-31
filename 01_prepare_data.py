import os.path as pth
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import argparse

def rgb_name_from_depth(d_file):
    dir, fname = pth.dirname(d_file), pth.basename(d_file)
    parent_dir, _ = pth.split(dir)
    fname_without_ext = pth.splitext(fname)[0]
    frame_number = fname_without_ext.split("-", maxsplit=1)[0]
    glob_pattern = pth.join(parent_dir, "rgb", "%s*.jpg" % frame_number)
    rgb_files = list(glob.glob(glob_pattern))

    return rgb_files[0] if rgb_files else False


def load_norm_scale_img(filepath, target_shape=(640 // 5, 480 // 5)):
    raw = cv2.imread(filepath)
    resized = cv2.resize(raw, target_shape)
    normalized = cv2.normalize(resized, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB)


def combine_depth_and_rgb(d_ary, rgb_ary):
    d_ary = cv2.cvtColor(d_ary, cv2.COLOR_BGR2GRAY)
    rgbd_ary = np.concatenate((rgb_ary, np.atleast_3d(d_ary)), axis=2)
    return rgbd_ary


def summarize_sample_set(set_path):
    rgbd_aries = list()
    for d_file in glob.glob("%s/depth/*.png" % set_path):
        rgb_file = rgb_name_from_depth(d_file)
        if not rgb_file:
            print("Unable to find rgb file for frame %s" % d_file)
            continue
        d_ary = load_norm_scale_img(d_file)
        rgb_ary = load_norm_scale_img(rgb_file)
        rgbd_aries.append(combine_depth_and_rgb(d_ary, rgb_ary))
        print("Done %s" % d_file)

    rgbd_ary = np.stack(rgbd_aries)
    np.savez(set_path, rgbd_ary=rgbd_ary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('set_parent', help='parent path containing sets (e.g. shoe, box)')
    args = parser.parse_args()

    for set_path in glob.glob(args.set_parent):
        if not pth.isdir(set_path):
            continue

        if pth.exists("%s.npz" % set_path):
            print("Output file for %s already exists, skipping" % set_path)
            continue

        summarize_sample_set(set_path)





