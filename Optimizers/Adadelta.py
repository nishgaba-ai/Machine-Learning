# AdadeltaOptimizer
with tf.name_scope('train'):
    train_step = tf.train.AdadeltaOptimizer().minimize(error)
    
init = tf.global_variables_initializer()
