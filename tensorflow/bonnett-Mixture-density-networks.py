from __future__ import division
from scipy.stats import beta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import seaborn as sns
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
from sklearn.preprocessing import RobustScaler
import tensorflow as tf

x = np.linspace(0.0, 1.0, 1000)
b0 = beta.pdf(x, 0.1 , 10)
b1 = beta.pdf(x, 50 , 7)
b2 = beta.pdf(x, 100 , 50)
b3 = beta.pdf(x, 10 , 10)
b4 = beta.pdf(x, 500 , 700)
plt.tight_layout()
plt.figure(num=None, figsize=(16, 4), dpi=80, facecolor='w', edgecolor='k')
plt.plot(x, b0, label='a=3, b=10', lw=7)
plt.plot(x, b1, label='a=50, b=7' ,lw=7)
plt.plot(x, b2, label='a=100, b=50', lw=7)
plt.plot(x, b3, label='a=10, b=50', lw=7)
plt.plot(x, b4, label='a=500, b=700', lw=7)
plt.legend(fontsize=20)
plt.title('Beta distribution', fontsize=30)

df_train = pd.read_csv('/tmp/galaxy_redshift_sims_train.csv', 
                       engine='c', na_filter=False, nrows=150000) 



fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
axes[0].set_xlim([0, 2.5])
axes[0].set_title('Redshift distrubtion (distance)', fontsize=18)
axes[0].legend(fontsize=13)
axes[0].set_xlabel('Redshift (distance)', fontsize=18)

axes[1].set_xlim([18, 32])
axes[1].set_title('Magnitude distributions (5 filters)', fontsize=18)
axes[1].legend(fontsize=13)
axes[1].set_xlabel('Magnitudes (higher == fainter)', fontsize=18)

axes[2].set_xlim([-6, 3])
axes[2].set_title('Distributions of the log-error', fontsize=18)
axes[2].legend(fontsize=11)
axes[2].set_xlabel('Log of error', fontsize=18)

df_train.head()

RS = RobustScaler()

### The training features ### 
feat_train = ['g', 'log_g_err', 'r', 'log_r_err', 'i', 'log_i_err',
              'z', 'log_z_err', 'y', 'log_y_err']

### The features for the validation set, ###  ###
### each galaxy has 5 distinct features 1 for each exposer time ###
feat_SN_1 = ['g_SN_1', 'log_g_err_SN_1', 'r_SN_1', 'log_r_err_SN_1',
             'i_SN_1', 'log_i_err_SN_1', 'z_SN_1', 'log_z_err_SN_1',
             'y_SN_1', 'log_y_err_SN_1']

feat_SN_2 = ['g_SN_2', 'log_g_err_SN_2', 'r_SN_2', 'log_r_err_SN_2',
             'i_SN_2', 'log_i_err_SN_2', 'z_SN_2', 'log_z_err_SN_2',
             'y_SN_2', 'log_y_err_SN_2']

feat_SN_3 = ['g_SN_3', 'log_g_err_SN_3', 'r_SN_3', 'log_r_err_SN_3',
             'i_SN_3', 'log_i_err_SN_3', 'z_SN_3', 'log_z_err_SN_3',
             'y_SN_3', 'log_y_err_SN_3']

feat_SN_4 = ['g_SN_4', 'log_g_err_SN_4', 'r_SN_4', 'log_r_err_SN_4',
             'i_SN_4', 'log_i_err_SN_4', 'z_SN_4', 'log_z_err_SN_4',
             'y_SN_4', 'log_y_err_SN_4']

feat_SN_5 = ['g_SN_5', 'log_g_err_SN_5', 'r_SN_5', 'log_r_err_SN_5',
             'i_SN_5', 'log_i_err_SN_5', 'z_SN_5', 'log_z_err_SN_5',
             'y_SN_5', 'log_y_err_SN_5']

###  training features with robust scaler ###
X_train = RS.fit_transform(df_train[feat_train])

### validation features in different noise levels ###
X_valid_SN_1 = RS.transform(df_valid[feat_SN_1])
X_valid_SN_2 = RS.transform(df_valid[feat_SN_2])
X_valid_SN_3 = RS.transform(df_valid[feat_SN_3])
X_valid_SN_4 = RS.transform(df_valid[feat_SN_4])
X_valid_SN_5 = RS.transform(df_valid[feat_SN_5])

### The targets that we wish to learn ###
Y_train = df_train['redshift']
Y_valid = df_valid['redshift']

### Some scaling of the target between 0 and 1 ###
### so we can model it with a beta function ###
### given that Beta function is not defined ###
### at 0 or 1 I've come up with this ulgy hack ###
max_train_Y = Y_train.max() + 0.00001
min_train_Y = Y_train.min() - 0.00001

### scaling : 0 < target < 1 ###
Y_train = (Y_train - min_train_Y) / (max_train_Y - min_train_Y)
Y_valid = (Y_valid - min_train_Y) / (max_train_Y - min_train_Y)

Y_train = Y_train[:, np.newaxis]  # add extra axis as tensorflow expects this 
Y_valid = Y_valid[:, np.newaxis] 

#
# MDN in TF
#

STDEV = 0.10
KMIX = 5  # number of mixtures
NOUT = KMIX * 3  # KMIX times a pi, alpha and beta

n_hidden_1 = 20  # 1st layer num neurons
n_hidden_2 = 20  # 2nd layer num neurons
n_hidden_3 = 10  # 2nd layer num neurons

# place holders
weight_decay = tf.placeholder(dtype=tf.float32, name="wd")
x = tf.placeholder(dtype=tf.float32, shape=[None, 10], name="x")
y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="y")

# hidden layer one 
Wh = tf.Variable(tf.random_normal([10, n_hidden_1], stddev=STDEV, dtype=tf.float32))
bh = tf.Variable(tf.random_normal([1, n_hidden_1], stddev=STDEV, dtype=tf.float32))

# hidden layer two 
Wh2 = tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=STDEV, dtype=tf.float32))
bh2 = tf.Variable(tf.random_normal([1, n_hidden_2], stddev=STDEV, dtype=tf.float32))

# hidden layer three
Wh3 = tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], stddev=STDEV, dtype=tf.float32))
bh3 = tf.Variable(tf.random_normal([1, n_hidden_3], stddev=STDEV, dtype=tf.float32))

# output layer
Wo = tf.Variable(tf.random_normal([n_hidden_3,NOUT], stddev=STDEV, dtype=tf.float32))
bo = tf.Variable(tf.random_normal([1, NOUT], stddev=STDEV, dtype=tf.float32))

## tanh acitivation function
hidden_layer1 = tf.nn.tanh(tf.add(tf.matmul(x, Wh), bh))
hidden_layer2 = tf.nn.tanh(tf.add(tf.matmul(hidden_layer1, Wh2),bh2))
hidden_layer3 = tf.nn.tanh(tf.add(tf.matmul(hidden_layer2, Wh3),bh3))

# # ReLU acitivation function
# hidden_layer1 = tf.nn.relu(tf.add(tf.matmul(x, Wh), bh))
# hidden_layer2 = tf.nn.relu(tf.add(tf.matmul(hidden_layer1, Wh2),bh2))
# hidden_layer3 = tf.nn.relu(tf.add(tf.matmul(hidden_layer2, Wh3),bh3))

output = tf.matmul(hidden_layer3, Wo) + bo
