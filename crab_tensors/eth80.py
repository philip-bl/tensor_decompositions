from libcrap import traverse_files # pip install libcrap

import random
from functools import lru_cache
import re

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from .feature_extraction import TuckerFeatureExtractor


class DictWithCounter(dict):
    """
    >>> d = DictWithCounter()
    >>> print(d.get_maybe_add("aaa"))
    0
    >>> print(d.get_maybe_add("bbb"))
    1
    >>> print(d.get_maybe_add("aaa"))
    0
    >>> print(sorted(d.items()))
    [('aaa', 0), ('bbb', 1)]
    """
    def __init__(self):
        super(DictWithCounter, self).__init__()
        self._next_value = 0
    
    def get_maybe_add(self, item):
        if item not in self:
            self[item] = self._next_value
            self._next_value += 1
        return self[item]


NUM_CLASSES = 8
NUM_OBJECTS_PER_CLASS = 10
NUM_OBJECTS = 8*10
NUM_ANGLES = 41
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
IMAGE_CHANNELS = 3


def fix_img_axes_for_show(image):
    return np.moveaxis(image, 0, 2)


def show_image(tensor, obj_id, angle_id, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    ax.imshow(fix_img_axes_for_show(tensor[obj_id, angle_id]))


class Eth80Dataset(object):
    def __init__(self, path):
        all_filenames = list(traverse_files(path))
        dataset = np.zeros(
            (NUM_OBJECTS, NUM_ANGLES, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH),
            dtype=np.float32, order="C"
        )
        object_classes = [None] * NUM_OBJECTS
        object_id_to_index = DictWithCounter()
        angles_to_index = DictWithCounter()
        loaded_num = 0
        for path in all_filenames:
            match = re.search(r"([a-z]+)(\d\d?)-(\d\d\d-\d\d\d).png", path)
            if match:
                object_id = match.group(1) + match.group(2)    
                angles = match.group(3)

                object_index = object_id_to_index.get_maybe_add(object_id)
                angles_index = angles_to_index.get_maybe_add(angles)
                object_classes[object_index] = match.group(1)

                image = plt.imread(path)
                dataset[object_index, angles_index] = np.moveaxis(image, 2, 0)
                loaded_num += 1
        if loaded_num != NUM_OBJECTS * NUM_ANGLES:
            raise ValueError(
                f"Loaded {loaded_num} objects, but should've loaded {NUM_OBJECTS * NUM_ANGLES}"
            )

        self.dataset_numpy = dataset
        self.object_classes = object_classes
        self.label_encoder = LabelEncoder()
        self.labels_numpy = self.label_encoder.fit_transform(object_classes)
        
        self._dataset_torch = None
        self._labels_torch = None
        
    @property
    def dataset_torch(self):
        import torch
        if self._dataset_torch is None:
            self._dataset_torch = torch.tensor(self.dataset_numpy)
        return self._dataset_torch

    @property
    def labels_torch(self):
        import torch
        if self._labels_torch is None:
            self._labels_torch = torch.tensor(self.labels_numpy)
        return self._labels_torch

    def stratified_split(self, num_test_per_class, use_torch):
        assert 1 <= num_test_per_class < NUM_OBJECTS_PER_CLASS
        if not use_torch:
            raise NotImplementedError("stratified split is not implemented for numpy")
        import torch
        obj_indices_sorted_by_class = torch.argsort(self.labels_torch)
        test_objects = set()
        for label in range(NUM_CLASSES):
            obj_indices_in_class = random.choices(
                range(NUM_OBJECTS_PER_CLASS), k=num_test_per_class
            )
            new_test_objects = obj_indices_sorted_by_class[[
                label*10 + ind_in_class for ind_in_class in obj_indices_in_class
            ]]
            test_objects.update(x.item() for x in new_test_objects)
        train_objects = sorted(frozenset(range(NUM_OBJECTS)) - test_objects)
        test_objects = sorted(test_objects)
        X_train = self.dataset_torch[train_objects]
        y_train = self.labels_torch[train_objects]
        X_test = self.dataset_torch[test_objects]
        y_test = self.labels_torch[test_objects]
        return X_train, y_train, X_test, y_test
    
    def choose_random_image(self):
        return random.randint(0, NUM_OBJECTS-1), random.randint(0, NUM_ANGLES-1)
    
    def get_random_image(self, use_torch):
        indices = self.choose_random_image()
        if use_torch:
            return self.dataset_torch[indices]
        else:
            return self.dataset_numpy[indices]
    
    def show_random_image(self):
        show_image(self.dataset_numpy, *self.choose_random_image())


def extract_X_y_train_test(
    dataset, num_test_objects_per_class, extracted_features_shape,
    return_torch_datasets=False
):
    """Performs stratified split of ETH80, does feature extraction via
    Tucker decomposition. Returns X_train, y_train, X_test, y_test."""
    
    tensor_train, y_train, tensor_test, y_test = dataset.stratified_split(
        num_test_objects_per_class, use_torch=True
    )
    
    extractor = TuckerFeatureExtractor(
        tensor_train.shape[1:],
        extracted_features_shape
    )
    core_train = extractor.fit_transform(tensor_train)
    core_test = extractor.transform(tensor_test)
    
    X_train = core_train.reshape(core_train.shape[0], -1)
    X_test = core_test.reshape(core_test.shape[0], -1)
    assert X_train.shape[1] == X_test.shape[1]
    
    if not return_torch_datasets:
        return X_train, y_train, X_test, y_test
    from torch.utils.data import TensorDataset
    return TensorDataset(X_train, y_train), TensorDataset(X_test, y_test)
