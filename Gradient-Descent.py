#GradientDescentOptimizer
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(learning_rate=0.01, name='GradientDescent').minimize(error)
    
init = tf.global_variables_initializer()
