import tensorflow as tf
import os.path
from Mnist_train.mnist_build_network import build_network
from tensorflow.python.framework import graph_util


def create_inference_graph():
    """Build the mnist model for evaluation."""
# Create an output to use for inference.
    logits = build_network(is_training=False)   # is_training=False
    tf.nn.softmax(logits, name='output')


def load_variables_from_checkpoint(sess, start_checkpoint):
    """Utility function to centralize checkpoint restoration.

    Args:
      sess: TensorFlow session.
      start_checkpoint: Path to saved checkpoint on disk.
    """
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, start_checkpoint)


def main():
    # Create the model and load its weights.
    # sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        create_inference_graph()  # is_training=False
        # if FLAGS.quantize:
        tf.contrib.quantize.create_eval_graph()
        load_variables_from_checkpoint(sess, '/home/angela/tensorflow/tensorflow/Mnist_train/mnist_fakequantize.ckpt')
        # Turn all the variables into inline constants inside the graph and save it.
        frozen_graph_def = graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['output'])
        tf.train.write_graph(
            frozen_graph_def,
            os.path.dirname('/home/angela/tensorflow/tensorflow/Mnist_train/mnist_frozen_graph.pb'),
            os.path.basename('/home/angela/tensorflow/tensorflow/Mnist_train/mnist_frozen_graph.pb'),
            as_text=False)
        tf.logging.info('Saved frozen graph to %s', '/home/angela/tensorflow/tensorflow/Mnist_train/mnist_frozen_graph.pb')


main()
