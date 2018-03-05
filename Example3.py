import tensorflow as tf
"""
Session会话 的两种打开方式
"""
# 两个矩阵 一行两列 一列两行
matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],
                      [2]])

product = tf.matmul(matrix1,matrix2) # matrix multiply 类似 np.dot(m1,m2)

# method 1
# sess = tf.Session()
# result = sess.run(product)
# print(result)
# sess.close()

# method 2
with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)