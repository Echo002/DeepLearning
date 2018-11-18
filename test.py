import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE = 784
OUTPUT_NODE = 10

LAYER1_NODE = 500  # 隐藏层的节点数（本结构只有一个隐藏层）

BATCH_SIZE = 100

LRARNING_RATE_BASE = 0.8
LRARNING_RATE_DECAY = 0.99  # 学习率的衰减率

REGULARIZATION_RATE = 0.0001  # 正则化项在损失函数中的系数

TRAINNING_STEPS = 30000  # 训练轮数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率


def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)  # 计算隐层的前向传播结果
        return tf.matmul(layer1, weights2) + biases2
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)


# 训练模型过程
def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

    # 生成隐藏层的参数
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

    # 生成输出层的数据
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 计算当前参数前向传播的结果。这里给出的用于计算平均滑动的类为 NONE,所以函数不会使用参数的滑动平均值
    y = inference(x, None, weights1, biases1, weights2, biases2)
    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)

    # 计算交叉熵                (需要将原文的语句做以下修改)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.arg_max(y_, 1), logits=y)
    # 计算当前batch中所有样例的交叉熵的平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # 计算模型的正则化损失
    regularization = regularizer(weights1) + regularizer(weights2)

    # 总损失等于交叉熵损失和正则化损失的和
    loss = cross_entropy_mean + regularization
    # 设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(LRARNING_RATE_BASE, global_step, mnist.train.num_examples / BATCH_SIZE,
                                               LRARNING_RATE_DECAY)  # 学习率衰减速度
    # 优化损失函数，这里的损失函数包括交叉熵损失函数和L2损失函数

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 一次进行反向传播参数更新网络中的 参数和参数的滑动平均值
    # train_op=tf.group(train_step,variable_averages_op)     #此句话下面的两句话是等价操作
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')
    correct_prediction = tf.equal(tf.arg_max(average_y, 1), tf.arg_max(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 初始化会话并开始训练过程
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  # 初始化所有的变量
        validate_feed = {x: mnist.validation.images,
                         y_: mnist.validation.labels}
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        # 迭代的训练神经网络
        for i in range(TRAINNING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d trainingstep(s), validation accuracy using average model is %g" % (i, validate_acc))

            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("After %d training step(s),test accuracy using average model is %g" % (TRAINNING_STEPS, test_acc))


# 主程序入口
def main(argv=None):
    mnist = input_data.read_data_sets("E:\DataSet\MNIST", one_hot=True)
    train(mnist)

if __name__ == "__main__":
    tf.app.run()