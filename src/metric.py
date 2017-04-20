import tensorflow as tf
import numpy as np


def accuracy_gpu(prediction, target):
    pred_label = tf.argmax(prediction, axis=1)
    target_label = tf.argmax(target, axis=1)
    counts = tf.to_float(tf.equal(pred_label, target_label))
    return tf.reduce_mean(counts)


def accuracy_cpu(prediction, target):
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.argmax(target, axis=1)
    return np.mean(pred_label==target_label)