
# coding: utf-8

# # MNIST with Uncertainty

# In[1]:

import numpy as np
import tensorflow as tf


# ## Goal: Find $p(\mathbf{y}^*|\mathbf{x}^*, \mathbf{X}, \mathbf{Y})=\int{p(\mathbf{y}^*|\mathbf{x}^*, \mathbf{w}) p(\mathbf{w}|\mathbf{X}, \mathbf{Y})d\mathbf{w}}$
# ### Variational Inference
# ### $$q(\mathbf{y}^*|\mathbf{x}^*)=\int{p(\mathbf{y}^*|\mathbf{x}^*, \mathbf{w}) q(\mathbf{w})d\mathbf{w}}$$

# ## Uncertainties
# ## Let $y=f(\mathbf{x})+\epsilon$ where $\epsilon \sim \mathcal{N}(0, \sigma_a^2)$
# ### $$
# \begin{align*}
# \mathbb{E}[y-\hat{f}(\mathbf{x})]^2
# &=\mathbb{E}[y-f(\mathbf{x})+f(\mathbf{x})-\hat{f}(\mathbf{x})]^2\\
# &=\mathbb{E}[\epsilon+f(\mathbf{x})-\hat{f}(\mathbf{x})]^2\\
# &=\mathbb{E}[\epsilon]^2+\mathbb{E}[f(\mathbf{x})-\hat{f}(\mathbf{x})]^2\\
# &=\sigma_a^2 + \mathbb{E}[f(\mathbf{x})-\hat{f}(\mathbf{x})]^2\\
# &=\sigma_a^2 + \sigma_e^2\\
# \end{align*}
# $$
# 
# ### And, $$
# \begin{align*}
# \sigma_e^2 &= \mathbb{E}[f(\mathbf{x})-\hat{f}(\mathbf{x})]^2\\
# &=\mathbb{E}[f(\mathbf{x})-\bar{f}(\mathbf{x})+\bar{f}(\mathbf{x})-\hat{f}(\mathbf{x})]^2\\
# &=\mathbb{E}[f(\mathbf{x})-\bar{f}(\mathbf{x})]^2+\mathbb{E}[\bar{f}(\mathbf{x})-\hat{f}(\mathbf{x})]^2\\
# &=bias(\hat{f}(\mathbf{x}))^2+Var(\hat{f}(\mathbf{x}))\\\\
# &\text{Assuming $\hat{f}(\mathbf{x})$ is a unbiased estimator, we have}\\
# &=Var(\hat{f}(\mathbf{x}))\\
# &=\mathbb{E}[\hat{f}(\mathbf{x})^2]-\mathbb{E}[\hat{f}(\mathbf{x})]^2
# \end{align*}
# $$

# ### Hence, $$
# \begin{align*}
# \mathbb{E}[y-\hat{f}(\mathbf{x})]^2
# &=\sigma_a^2 + \sigma_e^2\\
# &=\sigma_a^2 + \mathbb{E}[\hat{f}(\mathbf{x})^2]-\mathbb{E}[\hat{f}(\mathbf{x})]^2
# \end{align*}
# $$

# ### We will estimate the uncertainties using MC Dropout as follows $$
# \begin{align*}
# \mathbb{E}[y-\hat{f}(\mathbf{x})]^2
# &=\sigma_a^2 + \sigma_e^2\\
# &=\sigma_a^2 + \mathbb{E}[\hat{f}(\mathbf{x})^2]-\mathbb{E}[\hat{f}(\mathbf{x})]^2\\
# &\approx\hat{\sigma_a}^2 + \frac{1}{T}\sum_{t=1}^{T}{\hat{f_t}(\mathbf{x})^2}-\bigg(\frac{1}{T}\sum_{t=1}^{T}{\hat{f_t}(\mathbf{x})\bigg)}^2
# \end{align*}
# $$
# 
# ### Sampling $$\mathbf{w}_t\sim q(\mathbf{w})$$

# In[2]:

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

train_img = mnist.train._images
train_label = mnist.train._labels.copy()
test_img = mnist.test._images
test_label = mnist.test._labels


# ### Unseen label or Noise label

# In[ ]:

# 8 제거
# not_8 = np.where(train_label != 8)[0]
#
# train_img = train_img[not_8, :]
# train_label = train_label[not_8]

# 8에 노이즈
# idx_8 = np.where(train_label == 8)[0]
# train_label[idx_8] = np.random.choice(np.arange(10), idx_8.shape[0])


# ### Hyperparameter

# In[3]:

bn_moment = 0.9
n_mc_dropout = 50 # Dropout의 횟수

batch_size_tr = 32
batch_size_ts = 64
model_name = 'MNIST_Uncertainty'


# ### Weight initializer & Regularizer

# In[7]:

weight_initializer = tf.contrib.layers.variance_scaling_initializer()
regularizer = tf.contrib.layers.l2_regularizer(10**-4)


# In[8]:

x = tf.placeholder(tf.float32, [None, 784], name='x')
y = tf.placeholder(tf.int32, [None], name='target')

training = tf.placeholder(tf.bool, name='training')
conv_keep_prob = tf.placeholder(tf.float32, name='conv_keep_prob')
fc_keep_prob = tf.placeholder(tf.float32, name='fc_keep_prob')


# ### Copy input data for MC integration

# In[9]:

repeat = tf.transpose(tf.tile(tf.expand_dims(x, 0), [n_mc_dropout, 1, 1]), [1, 0, 2])
x_img = tf.reshape(repeat, [-1, 28, 28, 1])

y_repeat = tf.transpose(tf.tile(tf.expand_dims(y, 0), [n_mc_dropout, 1]))
y_repeat = tf.reshape(y_repeat, [-1])


# ## Network architecture

# ### Convolution Layers

# In[10]:

with tf.variable_scope('conv1'):
    net = tf.layers.conv2d(x_img, 32, kernel_size=[3, 3], strides=[2, 2], padding='SAME',
                           kernel_initializer=weight_initializer,
                           kernel_regularizer=regularizer, bias_regularizer=regularizer)
    net = tf.layers.batch_normalization(net, momentum=bn_moment, training=training)
    net = tf.nn.relu(net)
    net = tf.layers.dropout(net, conv_keep_prob, training=training)

with tf.variable_scope('conv2'):
    net = tf.layers.conv2d(net, 64, kernel_size=[3, 3], padding='SAME',
                           kernel_initializer=weight_initializer,
                           kernel_regularizer=regularizer, bias_regularizer=regularizer)
    net = tf.layers.batch_normalization(net, momentum=bn_moment, training=training)
    net = tf.nn.relu(net)
    net = tf.layers.dropout(net, conv_keep_prob, training=training)

with tf.variable_scope('conv3'):
    net = tf.layers.conv2d(net, 64, kernel_size=[3, 3], padding='SAME',
                           kernel_initializer=weight_initializer,
                           kernel_regularizer=regularizer, bias_regularizer=regularizer)
    net = tf.layers.batch_normalization(net, momentum=bn_moment, training=training)
    net = tf.nn.relu(net)
    net = tf.layers.dropout(net, conv_keep_prob, training=training)

with tf.variable_scope('conv4'):
    net = tf.layers.conv2d(net, 128, kernel_size=[3, 3], strides=[2, 2], padding='SAME',
                           kernel_initializer=weight_initializer,
                           kernel_regularizer=regularizer, bias_regularizer=regularizer)
    net = tf.layers.batch_normalization(net, momentum=bn_moment, training=training)
    net = tf.nn.relu(net)
    net = tf.layers.dropout(net, conv_keep_prob, training=training)


# ### Fully Connected Layers

# In[ ]:

with tf.variable_scope('fc1'):
    net = tf.layers.conv2d(net, 512, kernel_size=[7, 7], padding='VALID',
                           kernel_initializer=weight_initializer,
                           kernel_regularizer=regularizer, bias_regularizer=regularizer)
    net = tf.layers.batch_normalization(net, momentum=bn_moment, training=training)
    net = tf.nn.relu(net)
    net = tf.layers.dropout(net, fc_keep_prob, training=training)

with tf.variable_scope('fc2'):
    net = tf.layers.conv2d(net, 512, kernel_size=[1, 1], padding='VALID',
                           kernel_initializer=weight_initializer,
                           kernel_regularizer=regularizer, bias_regularizer=regularizer)
    net = tf.layers.batch_normalization(net, momentum=bn_moment, training=training)
    net = tf.nn.relu(net)
    net = tf.layers.dropout(net, fc_keep_prob, training=training)


# ### Noise sampling

# In[ ]:

with tf.variable_scope('logit'):
    net = tf.layers.conv2d(net, 11, kernel_size=[1, 1], padding='VALID',
                           kernel_initializer=weight_initializer,
                           kernel_regularizer=regularizer, bias_regularizer=regularizer)
    net = net[:, 0, 0, :]
    
    mu = net[..., :10]
    s = net[..., -1]
    sigma = tf.exp(s/2)
    target_shape = tf.shape(mu)
    
    mc_out = mu + tf.random_normal(target_shape)*tf.expand_dims(sigma, -1)


# ### Compute Loss

# According to [the paper](https://arxiv.org/abs/1703.04977), loss has to be calculated as follow.
# \begin{align*}
# p(y=c|\mathbf{x}, \mathbf{X}, \mathbf{Y})
# &=\int{Softmax(f_\mathbf{w}(\mathbf{x}))p(\mathbf{w}|\mathbf{X}, \mathbf{Y})d\mathbf{w}}\\
# &\approx \int{Softmax(f_\mathbf{w}(\mathbf{x}))q(\mathbf{w})d\mathbf{w}}\\
# &\approx \frac{1}{T}\sum_{t=1}^{T}{Softmax(f_{\mathbf{w}_t}(\mathbf{x}))}
# \end{align*}
# Sampling $$\mathbf{w}_t \sim q(\mathbf{w})$$

# Hence, $$
# \begin{align*}
# \mathcal{L}=-\sum_i{\log{\frac{1}{T}\sum_t{Softmax\big(f_{\mathbf{w}_t}(\mathbf{x})\big)}}}
# \end{align*}
# $$

# But, following code minimize
# $$
# \mathcal{L^{alt}}=-\sum_i{\frac{1}{T}\sum_t{\log{Softmax\big(f_{\mathbf{w}_t}(\mathbf{x})\big)}}}
# $$
# with weight decay.

# In[11]:

with tf.variable_scope('loss'):
    # prob = tf.reduce_mean((tf.nn.softmax(mc_out), 0)

    # idx = tf.concat([tf.expand_dims(tf.range(target_shape[0]*target_shape[1]), -1), tf.expand_dims(y_repeat, -1)], -1)
    # loss = tf.gather_nd(tf.reduce_logsumexp(mc_out - tf.expand_dims(tf.reduce_logsumexp(mc_out, -1), -1), 1), idx)
    # loss -= tf.log(float(n_mc_dropout)) # 학습상 상수이기에 필요 없다.

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_repeat, logits=mc_out)

    loss = tf.reduce_mean(loss)
    tf.summary.scalar('cross_entropy', loss)

    loss += tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    tf.summary.scalar('total_loss', loss)


with tf.name_scope('optimizer'):
    global_step = tf.Variable(0, trainable=False, name='global_step')
    learning_rate = tf.placeholder(tf.float32)
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        rmsprop = tf.train.RMSPropOptimizer(learning_rate).minimize(loss, global_step=global_step)
        sgd = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)


# ### Compute Uncertainties
# \begin{align*}
# \mathbb{E}[y-\hat{f}(\mathbf{x})]^2
# &=\sigma_a^2 + \sigma_e^2\\
# &=\sigma_a^2 + \mathbb{E}[\hat{f}(\mathbf{x})^2]-\mathbb{E}[\hat{f}(\mathbf{x})]^2\\
# &\approx\hat{\sigma_a}^2 + \frac{1}{T}\sum_{t=1}^{T}{\hat{f_t}(\mathbf{x})^2}-\bigg(\frac{1}{T}\sum_{t=1}^{T}{\hat{f_t}(\mathbf{x})\bigg)}^2
# \end{align*}
# 

# In[ ]:

with tf.name_scope('uncertainties'):
    mu = tf.reshape(mu, [-1, 50, 10])
    sigma = tf.reshape(sigma, [-1, 50])
    epistemic = tf.reduce_mean(mu**2, 1) - tf.reduce_mean(mu, 1)**2
    aleatoric = tf.reduce_mean(sigma**2, 1)


# \begin{align*}
# p(y=c|\mathbf{x}, \mathbf{X}, \mathbf{Y})
# &=\int{Softmax(f_\mathbf{w}(\mathbf{x}))p(\mathbf{w}|\mathbf{X}, \mathbf{Y})d\mathbf{w}}\\
# &\approx \int{Softmax(f_\mathbf{w}(\mathbf{x}))q(\mathbf{w})d\mathbf{w}}\\
# &\approx \frac{1}{T}\sum_{t=1}^{T}{Softmax(f_{\mathbf{w}_t}(\mathbf{x}))}
# \end{align*}
# Sampling $$\mathbf{w}_t \sim q(\mathbf{w})$$

# In[ ]:

with tf.name_scope('y_prob'):
    # expected_prob = tf.reduce_mean(tf.exp(mc_out - tf.expand_dims(tf.reduce_logsumexp(mc_out, -1), -1)), 1)

    mc_out_reshape = tf.reshape(mc_out, [-1, 50, 10])
    expected_prob = tf.reduce_mean(tf.exp(mc_out_reshape - tf.expand_dims(tf.reduce_logsumexp(mc_out_reshape, -1), -1)), 1)

    y_pred = tf.argmax(expected_prob, 1, output_type=tf.int32)
    tf.summary.histogram('probability', expected_prob)
    correct_prediction = tf.equal(y_pred, y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)


# ### Training

# In[ ]:

merged = tf.summary.merge_all()

def feed_dict(train, batch_size, lr=0.01):
    if train:
        batch_idx = np.random.choice(train_img.shape[0], batch_size, False)
        xs = train_img[batch_idx, :]
        ys = train_label[batch_idx]
        tr = True
    else:
        batch_idx = np.random.choice(test_img.shape[0], batch_size, False)
        xs = test_img[batch_idx, :]
        ys = test_label[batch_idx]
        tr = True
    return {x: xs, y: ys, training: tr, learning_rate: lr, conv_keep_prob: 0.8, fc_keep_prob: 0.8}



config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())


train_writer = tf.summary.FileWriter(model_name + '/train',sess.graph)
test_writer = tf.summary.FileWriter(model_name + '/test')

optimizer = rmsprop


# In[ ]:

i = 1

lr = 0.01
while (i <= (3000)):
    if i % 10 == 0:
        summary, tr = sess.run([merged, optimizer], feed_dict=feed_dict(True, batch_size_tr, lr))
        train_writer.add_summary(summary, i)
        summary, acc, bat_loss = sess.run([merged, accuracy, loss], feed_dict=feed_dict(False, batch_size_ts))
        test_writer.add_summary(summary, i)
        print('Accuracy and loss at step %s: %s %s' % (i, acc, bat_loss))
    else:
        tr = sess.run(optimizer, feed_dict=feed_dict(True, batch_size_tr, lr))
    i += 1

