import os
import tensorflow as tf
import time
import datetime


__all__ = ["MnistDNN"]


class MnistDNN:

    LOGDIR = "/Users/zy/Documents/workspace/mnist-tensorflow/LOG/"

    def __init__(self,
                 hidden_layers=(100, 30),
                 learning_rate=1e-3):
        self._hidden_layers = hidden_layers
        self._learning_rate = learning_rate

        self._session = None

        self._x = None
        self._y = None

        self._summary = None

        self._train_op = None
        self._accuracy_op = None

    @classmethod
    def get_data(cls):
        cls.mnist = tf.contrib.learn.datasets.mnist.read_data_sets(
            train_dir=cls.LOGDIR + "data", one_hot=True)

    def build(self):
        tf.reset_default_graph()
        self._session = tf.Session()

        # Setup placeholders, and reshape the data
        self._x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
        x_image = tf.reshape(self._x, [-1, 28, 28, 1])
        tf.summary.image('input', x_image, 3)

        self._y = tf.placeholder(tf.float32, shape=[None, 10], name="labels")

        activation = self._x
        pre_num_cells = 784

        nth = 0
        for nth, num_cells in enumerate(self._hidden_layers):
            layer_name = "fc_%d" % nth
            activation = self.fc_layer(activation, pre_num_cells, num_cells, layer_name)
            pre_num_cells = num_cells

        layer_name = "fc_%d" % (nth + 1)
        logit = self.weighted_input(activation, pre_num_cells, 10, layer_name)
        # logit = tf.nn.relu(fc)
        tf.summary.histogram("%s/relu" % layer_name, logit)

        with tf.name_scope("cross_entropy"):
            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=logit, labels=self._y),
                name="cross_entropy")
            tf.summary.scalar("cross_entropy", cross_entropy)

        with tf.name_scope("train"):
            optimizer = tf.train.AdamOptimizer(self._learning_rate)
            self._train_op = optimizer.minimize(cross_entropy)

        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(tf.argmax(logit, 1), tf.argmax(self._y, 1))
            self._accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar("accuracy", self._accuracy_op)

    def run(self, num_epochs=2001, batch_size=100):
        self._summary = tf.summary.merge_all()

        saver = tf.train.Saver()

        self._session.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(self.LOGDIR + self.get_hparam())
        writer.add_graph(self._session.graph)

        for i in range(num_epochs):
            xs, ys = self.mnist.train.next_batch(batch_size)
            if i % 5 == 0:
                [train_accuracy, s] = self._session.run(
                    [self._accuracy_op, self._summary],
                    feed_dict={self._x: xs, self._y: ys}
                )
                print("Train accuracy (%%): %.4f" % (train_accuracy * 100))
                writer.add_summary(s, i)

            if i % 500 == 0:
                saver.save(self._session, os.path.join(self.LOGDIR, "model.ckpt"), i)

            self._session.run(
                self._train_op,
                feed_dict={self._x: xs, self._y: ys}
            )

    @staticmethod
    def weighted_input(input, size_in, size_out, name="weighted_input"):
        with tf.name_scope(name):
            w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="Weights")
            b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="Biases")
            z = tf.matmul(input, w) + b
            tf.summary.histogram("weights", w)
            tf.summary.histogram("biases", b)
            tf.summary.histogram("weighted_input", z)

        return z

    def fc_layer(self, input, size_in, size_out, name="fc"):
        with tf.name_scope(name):
            z = self.weighted_input(input, size_in=size_in, size_out=size_out, name="weighted_input")
            activation = tf.nn.relu(z)
            tf.summary.histogram("relu", activation)
        return activation


    def get_hparam(self):
        dir_name = ""

        for l in self._hidden_layers:
            dir_name += "hl%d-"%l

        dir_name += "lr%e-" % self._learning_rate
        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H_%M_%S')

        dir_name += timestamp

        return dir_name
