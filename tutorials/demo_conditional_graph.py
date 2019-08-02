import random

import tensorflow as tf

import numpy as np

from demo_variable_length_batches import FixedLengthBatchIterator


class Model(object):
    def __init__(self, length=None, hidden_dim=10, input_dim=8, compose=None, projection=None, eager=None):
        super(Model, self).__init__()
        self.length = length
        self.compose = compose
        self.projection = projection
        self.eager = eager

    def __call__(self, x):
        # Project input to hidden input.
        model_input = self.projection(x)

        # Compose in seq.
        for i in range(length-1):
            if i == 0:
                left = tf.squeeze(tf.gather(model_input, axis=1, indices=[i]), axis=1)
            else:
                left = h
            right = tf.squeeze(tf.gather(model_input, axis=1, indices=[i+1]), axis=1)
            h_input = tf.concat([left, right], axis=1)
            h = self.compose(h_input)

        output = h

        return {
            'output': output
        }


if __name__ == '__main__':
    tf.enable_eager_execution()

    random.seed(11)
    num_batches = 10  # This can be any number.
    tf_batch_size = 1  # Set to 1 since we batch manually.
    np_batch_size = 2  # This is the actually batch size.

    hidden_dim = 10
    input_dim = 8

    min_length, max_length = 2, 10
    lengths = [random.randint(min_length, max_length) for _ in range(100)]
    data = [np.random.randn(i, input_dim) for i in lengths]

    batch_iterator = FixedLengthBatchIterator(data, rng=np.random.RandomState(121), batch_size=np_batch_size)

    ds = tf.data.Dataset.from_generator(lambda: iter(batch_iterator), output_types=tf.float32, output_shapes=(None, None, input_dim),)

    tf_graph = tf.Graph()
    with tf_graph.as_default():
        projection = tf.keras.layers.Dense(hidden_dim)
        compose = tf.keras.layers.Dense(hidden_dim)
        model_table = {k: Model(length=k, compose=compose, projection=projection, hidden_dim=hidden_dim, input_dim=input_dim)
                       for k in range(min_length, max_length+1)}
        init = tf.global_variables_initializer()

    with tf.Session(graph=tf_graph) as sess:
        sess.run(init)

    for i, batch in enumerate(ds.repeat().batch(1).take(num_batches)):
        batch = tf.squeeze(batch, axis=0)
        length = batch.shape[1].value
        output = model_table[length](batch)['output']
