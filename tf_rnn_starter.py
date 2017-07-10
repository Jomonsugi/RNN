import tensorflow as tf
import numpy as np

def basic_rnn():
    '''
    placeholder nodes:
    we know that X0 and X1 will be a 2D tensor, with instances along the first dimension and features along the second dimension, and we know that the number of features is going to be equal to n_inputs, but we do not know yet how many instances each training batch will contain, thus None
    '''
    # TensorShape([Dimension(None), Dimension(3)])
    X0 = tf.placeholder(tf.float32, [None, n_inputs])
    # TensorShape([Dimension(None), Dimension(3)])
    X1 = tf.placeholder(tf.float32, [None, n_inputs])
    # TensorShape([Dimension(3), Dimension(5)])
    Wx = tf.Variable(tf.random_normal(shape=[n_inputs, n_neurons],dtype=tf.float32))
    # TensorShape([Dimension(5), Dimension(5)])
    Wy = tf.Variable(tf.random_normal(shape=[n_neurons, n_nuerons],dtype=tf.float32))
    # TensorShape([Dimension(1), Dimension(5)])
    b = tf.Variable(tf.zeros([1, n_neurons], dtype=tf.float32))

    # TensorShape([Dimension(None), Dimension(5)])
    Y0 = tf.tanh(tf.matmul(X0, Wx) + b)
    # TensorShape([Dimension(None), Dimension(5)])
    Y1 = tf.tanh(tf.matmul(Y0, Wy) + tf.matmul(X1, Wx) + b)

    init = tf.global_variables_initializer()

    X0_batch = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 0, 1]]) # t = 0
    X1_batch = np.array([[9, 8, 7], [0, 0, 0], [6, 5, 4], [3, 2, 1]]) # t = 1

    with tf.Session() as sess:
        init.run()
        Y0_val, Y1_val = sess.run([Y0, Y1], feed_dict={X0: X0_batch, X1: X1_batch})

# this function does the exact same thing as above with tf's built in rnn function
def basic_built_in():
    X0 = tf.placeholder(tf.float32, [None, n_inputs])
    X1 = tf.placeholder(tf.float32, [None, n_inputs])
    basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
    output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell, [X0, X1],
                                                    dtype=tf.float32)
    Y0, Y1 = output_seqs

    # now we are back to the exact same code as above, we just go here quicker
    init = tf.global_variables_initializer()

    X0_batch = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 0, 1]]) # t = 0
    X1_batch = np.array([[9, 8, 7], [0, 0, 0], [6, 5, 4], [3, 2, 1]]) # t = 1

    with tf.Session() as sess:
        init.run()
        Y0_val, Y1_val = sess.run([Y0, Y1], feed_dict={X0: X0_batch, X1: X1_batch})

# still doing the same thing except now with a while loop to run over the cell the appropriate number of times, thus relieving us from creating placeholder nodes for every X tensor
def dynamic_rnn():
    n_steps = 2
    #thus we only need one X
    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])

    basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
    outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

    init = tf.global_variables_initializer()
    #this is used for the dynamic_rnn as it will accept any arbitrary number of
    #of batches
    X_batch = np.array([
        [[0, 1, 2], [9, 8, 7]], # instance 1
        [[3, 4, 5], [0, 0, 0]], # instance 2
        [[6, 7, 8], [6, 5, 4]], # instance 3
        [[9, 0, 1], [3, 2, 1]], # instance 4
    ])

    with tf.Session() as sess:
        init.run()
        outputs_val = outputs.eval(feed_dict={X: X_batch})

    return outputs_val

if __name__ == '__main__':
    tf.reset_default_graph()
    n_inputs = 3
    n_neurons = 5

    outputs_val = dynamic_rnn()
