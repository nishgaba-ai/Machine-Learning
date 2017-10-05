# L2 reg
with tf.name_scope('train'):
    train_step = tf.train.FtrlOptimizer(learning_rate=0.001,l2_regularization_strength=0.4).minimize(error)
    
init = tf.global_variables_initializer()
