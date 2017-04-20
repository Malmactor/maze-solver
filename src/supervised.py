import numpy as np
import tensorflow as tf
import conv_units as cu
import dataset_loader as loader
import metric


def build_graph(input_shape=(32, 49, 49, 6), global_config={}):

    # initialize variables
    returned_nodes = {}
    batch_size = input_shape[0]
    bn_moment = global_config["bn_momentum"]
    weight_decay = global_config["weight_decay"]
    learning_rate = global_config["learning_rate"]
    l1_regularizer = tf.contrib.layers.l1_regularizer(scale=weight_decay)

    # build place holders
    inputs = tf.placeholder("float32", input_shape)
    labels = tf.placeholder("int16", [batch_size, 4])
    training = tf.placeholder("bool", ())
    keep_prob = tf.placeholder("float32", ())
    returned_nodes.update({"input": inputs, "label": labels, "training": training, "keep_prob": keep_prob})

    # main structure
    # cnn base
    net = cu.branch4_avgpool_5x5(inputs, "layer1", training=training, bn_momentum=bn_moment, out_channel=32)
    net = cu.branch4_res_5x5(net, "layer2", training=training, bn_momentum=bn_moment)
    net = cu.branch3_maxpool_downsample_5x5(net, "layer3", training=training, bn_momentum=bn_moment)
    net = cu.branch3_res_7x7(net, "layer4", training=training, bn_momentum=bn_moment)
    net = cu.branch3_maxpool_downsample_9x9(net, "layer5", training=training, bn_momentum=bn_moment)
    net = cu.branch6_avgpool_5x5_downchannel(net, "layer6", training=training, bn_momentum=bn_moment, out_channel=42)

    # global average pooling and reshape
    net = cu.global_avg_dropout(net, "layer7", keep_prob=keep_prob)
    net = tf.reshape(net, [batch_size, -1])

    # fully connected
    net = cu.fc_dropout(net, 32, "layer8", training=training, bn_momentum=bn_moment, keep_prob=keep_prob)
    net = cu.fc_dropout(net, 4, "layer9", training=training, bn_momentum=bn_moment, keep_prob=keep_prob)

    # prediction and loss
    prediction = tf.nn.softmax(net)
    loss = tf.losses.softmax_cross_entropy(labels, net)

    # l1 regularization
    for var in tf.trainable_variables():
        if "weight" in var.name:
            loss += l1_regularizer(var)

    returned_nodes["prediction"] = prediction
    returned_nodes["loss"] = loss

    # metric node
    returned_nodes["accuracy"] = metric.accuracy_gpu(prediction, labels)

    # optimizer node
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    returned_nodes["train_op"] = train_op

    return returned_nodes


def train(global_config):

    # build tensorflow graph
    nodes = build_graph(global_config=global_config)

    # load dataset
    inputs, labels = loader.load_dataset()

    # start session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # training iterations
        for epoch in range(global_config["epoches"]):
            batch = loader.next_batch(global_config["batch_size"], inputs, labels)
            feeds = loader.create_feed(
                nodes, batch[0], batch[1], global_config["batch_size"], global_config["keep_prob"], True)

            accuracy, loss, _ = sess.run([nodes["accuracy"], nodes["loss"], nodes["train_op"]], feed_dict=feeds)

            if epoch % 20 == 1:
                pred = sess.run(nodes["prediction"], feed_dict=feeds)
                acc_cpu = metric.accuracy_cpu(pred, batch[1])
                print("epoch: {}, loss: {:.03}, acc_gpu: {}, acc_cpu: {}".format(epoch, loss, accuracy, acc_cpu))


if __name__ == '__main__':
    train(
        {"batch_size": 32,
         "keep_prob": 0.9,
         "bn_momentum": 0.95,
         "weight_decay": 0.0003,
         "learning_rate": 0.01,
         "epoches": 7000}
    )
