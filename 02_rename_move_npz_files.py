import os
import os.path as pth
import glob


if __name__ == '__main__':
    for npz_path in glob.glob("data/**/*.npz", recursive=True):
        npz_file = pth.basename(npz_path)
        data_dir, set_name = pth.split(pth.dirname(npz_path))
        new_path = pth.join(data_dir, "%s_%s" % (set_name, npz_file))

        print("%s -> %s" % (npz_path, new_path))
        os.rename(npz_path, new_path)

        # Afterwards clean up extracted files with
        # find . -type d -maxdepth 2 -mindepth 2 -exec rm -rf {} \;
