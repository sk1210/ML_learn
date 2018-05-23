import tensorflow as tf
import loadData

#----------------------------#



# start session and init variables
sess = tf.Session()
data = loadData.fashion()
init_var = tf.initialize_variables([data.iterator])
init = tf.global_variables_initializer()

sess.run(init_var)

# start training
sess.run(data.get_next())

