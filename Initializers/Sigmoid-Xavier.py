# SIGMOID XAVIER INTIALIZATION
def fully_connected_layer_sigmoidx(inputs, input_dim, output_dim, nonlinearity=tf.nn.sigmoid):
    weights = tf.Variable(
        tf.truncated_normal(
            [input_dim, output_dim], stddev=(2.6/(input_dim + output_dim))**0.5), 
        'weights')
    biases = tf.Variable(tf.zeros([output_dim]), 'biases')
    outputs = nonlinearity(tf.matmul(inputs, weights) + biases)
    return outputs
