import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from Mnist_train.mnist_build_network import build_network, x, y, keep_prob
# from Yolo_training_code.fuwuqi.vgg_fkqtz import build_det_hand_model

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def create_training_graph():
    g = tf.get_default_graph()
    logits = build_network(is_training=True)
    # logits = build_det_hand_model(input_shape=input_data, is_trainer=True)
    # Create the back propagation and training evaluation machinery in the graph.
    with tf.name_scope('cross_entropy'):
        cross_entropy_mean = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits))
        print('cost:', cross_entropy_mean)

    # if FLAGS.quantize:
    tf.contrib.quantize.create_training_graph(input_graph=g, quant_delay=0)
    optimize = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_mean)
    # optimize = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy_mean)

    prediction_labels = tf.argmax(logits, axis=1, name="output")
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    with tf.get_default_graph().name_scope('eval'):
        tf.summary.scalar('cross_entropy', cross_entropy_mean)
        tf.summary.scalar('accuracy', accuracy)

    return dict(
        x=x,
        y=y,
        keep_prob=keep_prob,
        optimize=optimize,
        cost=cross_entropy_mean,
        correct_prediction=correct_prediction,
        accuracy=accuracy,
    )


def train_network(graph):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(2000):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = sess.run([graph['accuracy']], feed_dict={
                                                                           graph['x']:batch[0],
                                                                           graph['y']:batch[1],
                                                                           graph['keep_prob']: 1.0})
                print("step %d, training accuracy %g"%(i, train_accuracy[0]))
            sess.run([graph['optimize']], feed_dict={
                                                       graph['x']:batch[0],
                                                       graph['y']:batch[1],
                                                       graph['keep_prob']:0.5})

        test_accuracy = sess.run([graph['accuracy']], feed_dict={
                                                                  graph['x']: mnist.test.images,
                                                                  graph['y']: mnist.test.labels,
                                                                  graph['keep_prob']: 1.0})
        print("Test accuracy %g" % test_accuracy[0])

        saver.save(sess, '/home/angela/tensorflow/tensorflow/Mnist_train/mnist_fakequantize.ckpt')
        tf.train.write_graph(sess.graph_def, '/home/angela/tensorflow/tensorflow/Mnist_train/', 'mnist_fakequantize.pbtxt', True)


def main():
    g1 = create_training_graph()
    train_network(g1)


main()








