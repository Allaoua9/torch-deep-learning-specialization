import h5py
import numpy as np
import torch


def load_data():
    train_dataset = h5py.File("datasets/train_catvnoncat.h5", "r")
    train_set_x_orig = torch.tensor(
        train_dataset["train_set_x"][:]
    )  # your train set features
    train_set_y_orig = torch.tensor(
        train_dataset["train_set_y"][:]
    )  # your train set labels

    test_dataset = h5py.File("datasets/test_catvnoncat.h5", "r")
    test_set_x_orig = torch.tensor(
        test_dataset["test_set_x"][:]
    )  # your test set features
    test_set_y_orig = torch.tensor(
        test_dataset["test_set_y"][:]
    )  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((train_set_y_orig.shape[0], 1)).to(
        torch.float
    )
    test_set_y_orig = test_set_y_orig.reshape((test_set_y_orig.shape[0], 1)).to(
        torch.float
    )

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
