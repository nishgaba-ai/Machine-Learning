# AdagradOptimizer
with tf.name_scope('train'):
    train_step = tf.train.AdagradOptimizer(learning_rate=0.01, name='Adagrad').minimize(error)
    
init = tf.global_variables_initializer()
