            
# FtrlOptimizer
with tf.name_scope('train'):
    train_step = tf.train.FtrlOptimizer(learning_rate=0.01).minimize(error)
    
init = tf.global_variables_initializer()
