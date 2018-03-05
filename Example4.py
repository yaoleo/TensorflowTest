import tensorflow as tf
"""
Variable变量 和python有所不同
"""
state = tf.Variable(0,name='counter')
#print(state.name)

one = tf.constant(1)

new_value = tf.add(state, one)
update = tf.assign(state,new_value)      # new_value 加载到state上

init = tf.global_variables_initializer() # 定义变量 一定有这步

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))