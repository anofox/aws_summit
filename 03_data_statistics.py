import glob

import collections
from showcase.image_sequence import create_rgbdseq_from_files

if __name__ == '__main__':
    seq = create_rgbdseq_from_files(glob_pattern="data/*.npz")
    print(len(seq))
    seq.plot_image_count_hist()

    # for b in seq:
    #    if isinstance(b, bool):
    #        break

    #    plt.figure()
    #    plt.imshow(b[0, :, :, 0:3])
    #    plt.show()
    #    print(b.shape)

    # 25000 Samples.
    # N x 96 x 128 x 4