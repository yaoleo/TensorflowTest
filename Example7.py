# tensorboard
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 定义添加神经层
def add_layer(inputs, in_size, out_size, n_layer, activation_funtion = None):
    layer_name = 'layer%s' % n_layer
    with tf.name_scope("layer"):
        with tf.name_scope("Weights"):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name="Weights")
            tf.summary.histogram(layer_name+'/weights',Weights)
        with tf.name_scope("biases"):
            biases  = tf.Variable(tf.zeros([1, out_size]) + 0.1) # bias 推荐值不是0 加0.1
            tf.summary.histogram(layer_name + '/biases', biases)
        with tf.name_scope("Wx_plus_b"):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases

        if activation_funtion is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_funtion(Wx_plus_b)
        tf.summary.histogram(layer_name + '/outputs', outputs)
        return outputs

x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# placeholder input
with tf.name_scope("inputs"):
    ys = tf.placeholder(tf.float32,[None,1], name='x_input')
    xs = tf.placeholder(tf.float32,[None,1], name='y_input')

# add hidden layer
l1 = add_layer(xs, 1, 10, n_layer = 1, activation_funtion= tf.nn.relu)
# add output layer
prediction = add_layer(l1, 10, 1, n_layer = 2, activation_funtion=None)

with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1])) # [1]按行求和 [0]按列求和
    tf.summary.scalar('loss', loss)
with tf.name_scope("train"):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
merged = tf.summary.merge_all()
#merged = tf.merge_all_summaries()
writer = tf.summary.FileWriter("logs/",sess.graph) # 加载到文件 从文件打开才能显示tensorboard

sess.run(init)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)


for i in range(1000):
    sess.run(train_step,feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        # print(sess.run(loss,feed_dict={xs: x_data, ys:y_data}))
        result = sess.run(merged, feed_dict={xs: x_data, ys: y_data})
        writer.add_summary(result, i)
