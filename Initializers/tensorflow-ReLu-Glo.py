inputs = tf.placeholder(tf.float32, [None, train_data.inputs.shape[1]], 'inputs')
targets = tf.placeholder(tf.float32, [None, train_data.num_classes], 'targets')
num_hidden = 500
num_hidden_1 = 500
num_hidden_2= 500

# RELU GLORIOT INITIALIZATION

with tf.name_scope('fc-layer-1'):
    hidden_1 = fully_connected_layer_relu(inputs, train_data.inputs.shape[1], num_hidden)
with tf.name_scope('fc-layer-2'):
    hidden_2 = fully_connected_layer_relu(hidden_1, num_hidden, num_hidden_1)
with tf.name_scope('fc-layer-2'):
    hidden_3 = fully_connected_layer_relu(hidden_2, num_hidden_1, num_hidden_2)
with tf.name_scope('output-layer'):
    outputs = fully_connected_layer_relu(hidden_3, num_hidden_2, train_data.num_classes, tf.identity)

with tf.name_scope('error'):
    error = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(outputs, targets))
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1)), 
            tf.float32))

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer().minimize(error)
    
init = tf.global_variables_initializer()



with tf.Session() as sess:
    sess.run(init)
    epo =[]
    epo_v=[0]
    acc_t_1=[]
    err_t_1=[]
    acc_v_1=[0]
    err_v_1=[0]
    ttrg=0
    for e in range(100):
        trg = time.time()
        running_error = 0.
        running_accuracy = 0.
        for input_batch, target_batch in train_data:
            _, batch_error, batch_acc = sess.run(
                [train_step, error, accuracy], 
                feed_dict={inputs: input_batch, targets: target_batch})
            running_error += batch_error
            running_accuracy += batch_acc
        running_error /= train_data.num_batches
        running_accuracy /= train_data.num_batches
        print('End of epoch {0:02d}: err(train)={1:.2f} acc(train)={2:.2f}'
              .format(e + 1, running_error, running_accuracy))
        ttrg += time.time()-trg
        epo.append(e+1)
        err_t_1.append(running_error)
        acc_t_1.append(running_accuracy)
        if (e + 1) % 5 == 0:
            epo_v.append(e+1)
            valid_error = 0.
            valid_accuracy = 0.
            for input_batch, target_batch in valid_data:
                batch_error, batch_acc = sess.run(
                    [error, accuracy], 
                    feed_dict={inputs: input_batch, targets: target_batch})
                valid_error += batch_error
                valid_accuracy += batch_acc
            valid_error /= valid_data.num_batches
            valid_accuracy /= valid_data.num_batches
            print('                 err(valid)={0:.2f} acc(valid)={1:.2f}'
                   .format(valid_error, valid_accuracy))
            err_v_1.append(valid_error)
            acc_v_1.append(valid_accuracy)
