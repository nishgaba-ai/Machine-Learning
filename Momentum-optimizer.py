    
# MomentumOptimizer
with tf.name_scope('train'):
    train_step = tf.train.MomentumOptimizer(learning_rate=0.01, momentum = 0.005).minimize(error)
    
init = tf.global_variables_initializer()
