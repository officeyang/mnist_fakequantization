import tensorflow as tf

x = tf.placeholder("float", shape=[None, 784], name='input')
y = tf.placeholder("float", shape=[None, 10], name='labels')
keep_prob = tf.placeholder("float", name='keep_prob')


def build_network(is_training):
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # convolution and pooling
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # convolution layer
    def lenet5_layer(layer, weight, bias):
        W_conv = weight_variable(weight)
        b_conv = bias_variable(bias)
        h_conv = conv2d(layer, W_conv) + b_conv
        return max_pool_2x2(h_conv)

    # connected layer
    def dense_layer(layer, weight, bias):
        W_fc = weight_variable(weight)
        b_fc = bias_variable(bias)
        return tf.matmul(layer, W_fc) + b_fc

    # first layer
    with tf.name_scope('first') as scope:
        x_image = tf.pad(tf.reshape(x, [-1,28,28,1]), [[0,0],[2,2],[2,2],[0,0]])
        firstlayer = lenet5_layer(x_image, [5,5,1,6], [6])

    # second layer
    with tf.name_scope('second') as scope:
        secondlayer = lenet5_layer(firstlayer, [5,5,6,16], [16])

    # third layer
    with tf.name_scope('third') as scope:
        W_conv3 = weight_variable([5,5,16,120])
        b_conv3 = bias_variable([120])
        thirdlayerconv = conv2d(secondlayer, W_conv3) + b_conv3
        thirdlayer = tf.reshape(thirdlayerconv, [-1,120])

    # dense layer1
    with tf.name_scope('dense1') as scope:
        dense_layer1 = dense_layer(thirdlayer, [120,84], [84])

    # dense layer2
    with tf.name_scope('dense2') as scope:
        dense_layer2 = dense_layer(dense_layer1, [84,10], [10])
    if is_training:
        finaloutput = tf.nn.softmax(tf.nn.dropout(dense_layer2, keep_prob), name="softmax")
    else:
        finaloutput = tf.nn.softmax(dense_layer2, name='softmax')
    print('finaloutput:', finaloutput)
    return finaloutput