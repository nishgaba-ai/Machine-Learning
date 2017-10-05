# TANH WITH RANDOM UNINFORM            
def fully_connected_layer_tanr(inputs, input_dim, output_dim, nonlinearity=tf.nn.sigmoid):
    weights = tf.Variable(
        tf.random_uniform(
            [input_dim, output_dim],minval=-1*(6./(input_dim+output_dim))**0.5, maxval= (6./(input_dim+output_dim))**0.5))
    biases = tf.Variable(tf.zeros([output_dim]), 'biases')
    outputs = nonlinearity(tf.matmul(inputs, weights) + biases)
    return outputs
