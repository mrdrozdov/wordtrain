import random

import numpy as np

import tensorflow as tf


class FixedLengthBatchIterator(object):
    def __init__(self, data, rng=None, batch_size=2, keep_partial=False):
        self.data = data
        self.rng = np.random.RandomState() if rng is None else rng
        self.batch_size = batch_size

        self.length_to_data = {}
        for x in self.data:
            self.length_to_data.setdefault(len(x), []).append(x)

        batches = []
        for length, lst in self.length_to_data.items():
            batches += [length] * (len(lst) // batch_size)
            if keep_partial and len(lst) % batch_size > 0:
                batches += [length]
        self.batches = batches

        self.order = None
        self.length_to_offset = None

    def reset(self):
        # Shuffle lengths.
        order = self.batches[:]
        self.rng.shuffle(order)
        self.order = order

        # Shuffle data.
        for length, lst in self.length_to_data.items():
            self.rng.shuffle(lst)

        # Reset offset.
        length_to_offset = {}
        for length, lst in self.length_to_data.items():
            length_to_offset[length] = 0
        self.length_to_offset = length_to_offset

    def __iter__(self):
        self.reset()

        for length in self.order:
            offset = self.length_to_offset[length]
            batch = self.length_to_data[length][offset:offset+self.batch_size]
            self.length_to_offset[length] += len(batch)
            yield batch

    def __len__(self):
        return len(self.order)


if __name__ == '__main__':
    tf.enable_eager_execution()

    random.seed(11)
    size = 8
    num_batches = 10  # This can be any number.
    tf_batch_size = 1  # Set to 1 since we batch manually.
    np_batch_size = 2  # This is the actually batch size.

    lengths = [random.randint(1, 10) for _ in range(100)]
    data = [np.random.randn(i, size) for i in lengths]

    batch_iterator = FixedLengthBatchIterator(data, rng=np.random.RandomState(121), batch_size=np_batch_size)

    ds = tf.data.Dataset.from_generator(lambda: iter(batch_iterator), output_types=tf.int32, output_shapes=(None, None, size),)

    for i, batch in enumerate(ds.repeat().batch(1).take(num_batches)):
        print(i, batch.shape)
