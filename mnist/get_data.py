#!/usr/bin/env python3

import sys
import numpy as np

def run():
    sys.stderr.write("Downloading data...\n")

    import tensorflow.examples.tutorials.mnist as tf_mnist
    data = tf_mnist.input_data.read_data_sets(
        "/tmp/mnist-download",
        validation_size=0,
        one_hot=True
    )

    sys.stderr.write("Saving data...\n")

    np.savez_compressed(
        "__mnist.npz",
        train_x=data.train.images.reshape(-1, 28, 28),
        train_y=data.train.labels,
        test_x=data.test.images.reshape(-1, 28, 28),
        test_y=data.test.labels
    )

if __name__ == "__main__":
    run()
