# L1 reg
with tf.name_scope('train'):
    train_step = tf.train.FtrlOptimizer(learning_rate=0.001,l1_regularization_strength=0.3).minimize(error)
    
init = tf.global_variables_initializer()
