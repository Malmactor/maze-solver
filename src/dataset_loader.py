import numpy as np


def load_dataset(filename=["inputs.npy", "labels.npy"]):
    return np.load(filename[0]), np.load(filename[1])


def next_batch(batch_size, inputs, labels):
    ids = np.random.choice(inputs.shape[0], size=batch_size)
    return inputs[ids, :, :, :], labels[ids, :]


def create_feed(nodes, inputs, labels, batch_size, keep_prob, training):
    batch = next_batch(batch_size, inputs, labels)
    return {
        nodes["input"]: batch[0],
        nodes["label"]: batch[1],
        nodes["training"]: training,
        nodes["keep_prob"]: keep_prob}
