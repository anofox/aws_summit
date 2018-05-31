import glob
import os.path as pth
import numpy as np
import collections
import matplotlib.pylab as plt
import keras.utils


class RgbdSetElement(object):
    def __init__(self, npz_path):
        self.npz_path = npz_path
        self.npz_file = pth.basename(npz_path)
        self.set_name = self.npz_file.split("_", maxsplit=1)[0]
        self.image_cnt = self.__image_count_from_file(npz_path)
        self.current_rbgd_ary = False

    def __image_count_from_file(self, npz_path):
        with np.load(npz_path) as f:
            rgbd_ary = f["rgbd_ary"]
            return rgbd_ary.shape[0]

    def read_next_batch(self, batch_size):
        if isinstance(self.current_rbgd_ary, bool) and not self.current_rbgd_ary:
            f = np.load(self.npz_path)
            self.current_rbgd_ary = f["rgbd_ary"]
            self.current_batch_idx = 0
        try:
            current_batch = self.current_rbgd_ary[self.current_batch_idx*batch_size:(self.current_batch_idx + 1)*batch_size, :, :, :]
            if (current_batch.shape[0] < batch_size):
                raise ValueError()

            self.current_batch_idx += 1
            return current_batch
        except:
            self.current_rbgd_ary = False
            self.current_batch_idx = False
            return False


class RgbdSet(object):
    def __init__(self):
        self.set_name = False
        self.elements = list()

    @property
    def image_count(self):
        if not self.set_name or not self.elements:
            raise ValueError("Invalid state, set has no name or no elements")

        return sum([e.image_cnt for e in self.elements])

    def append(self, set_el):
        if not self.set_name:
            self.set_name = set_el.set_name

        if self.set_name != set_el.set_name:
            raise ValueError("Set name and element set name does not match")

        self.elements.append(set_el)
        return self


class RgbdSetSequence(keras.utils.Sequence):
    def __init__(self, rgbd_set:RgbdSet, batch_size, validation_split=0.2, is_validation=False, max_len=25000):
        self.rgbd_set = rgbd_set
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.is_validation = is_validation
        self.max_len = max_len
        self.image_cnt = self.__calculate_image_count()
        self.element_queue = self.__copy_elements(rgbd_set, validation_split, is_validation)

    def __calculate_image_count(self):
        N = min(self.rgbd_set.image_count, self.max_len)
        if self.is_validation:
            return N * self.validation_split
        else:
            return N * (1. - self.validation_split)

    def __len__(self):
        return int(np.ceil(self.image_cnt / float(self.batch_size)))

    @property
    def current_element(self):
        return self.element_queue[0] if self.element_queue else False

    def __getitem__(self, idx):
        if not self.current_element:
            return False

        next_batch = self.current_element.read_next_batch(self.batch_size)
        if isinstance(next_batch, bool) and next_batch == False:
            self.element_queue.pop()
            return self.__getitem__(idx)

        return next_batch

    def __copy_elements(self, rgbd_set, validation_split, is_validation):
        all_els = reversed(rgbd_set.elements) if is_validation else rgbd_set.elements
        cpy_els = []
        M = 0
        for el in all_els:
            if M >= self.image_cnt: break
            M += el.image_cnt
            cpy_els.append(el)

        return collections.deque(cpy_els)


class RgbdSeq(keras.utils.Sequence):
    def __init__(self, all_sets_dict, batch_size, validation_split=0.2, is_validation=False, max_per_set_len=25000):
        self.sets_dict = all_sets_dict
        self.seqs_dict = dict([(k, RgbdSetSequence(s, batch_size, validation_split, is_validation, max_per_set_len))
                              for k, s in all_sets_dict.items()])

    def __len__(self):
        return sum(len(s) for s in self.seqs_dict.values())

    def __getitem__(self, idx):
        seq_idx = idx % len(self.seqs_dict)
        seq_els = list(self.seqs_dict.values())
        return seq_els[seq_idx][idx]

    def plot_image_count_hist(self):
        plt.figure()
        plt.bar(self.sets_dict.keys(), [v.image_count for v in self.sets_dict.values()])
        plt.show()


def create_rgbdseq_from_files(glob_pattern="data/*.npz", is_validation=False, batch_size=250):
    rgbd_sets = collections.defaultdict(RgbdSet)
    for npz_path in glob.glob(glob_pattern):
        print("Loading %s" % npz_path)
        set_el = RgbdSetElement(npz_path)
        rgbd_sets[set_el.set_name].append(set_el)

    seq = RgbdSeq(rgbd_sets, is_validation=is_validation, batch_size=batch_size)
    return seq