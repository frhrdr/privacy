import tensorflow as tf
import numpy as np
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

file_name='models_fmnist/private_lr025/model.ckpt-3510'
print_tensors_in_checkpoint_file(file_name=file_name, tensor_name='', all_tensors=False)

c1b = tf.get_variable("conv2d/bias", shape=[16])
c1k = tf.get_variable("conv2d/kernel", shape=[8, 8, 1, 16])
c2b = tf.get_variable("conv2d_1/bias", shape=[32])
c2k = tf.get_variable("conv2d_1/kernel", shape=[4, 4, 16, 32])

d1b = tf.get_variable("dense/bias", shape=[32])
d1k = tf.get_variable("dense/kernel", shape=[512, 32])
d2b = tf.get_variable("dense_1/bias", shape=[10])
d2k = tf.get_variable("dense_1/kernel", shape=[32, 10])

saver = tf.train.Saver()

with tf.Session() as sess:
  # saver.restore(sess, "models_fmnist/nonprivate_lr005/model.ckpt-14040")
  saver.restore(sess, file_name)
  c1b_mat = c1b.eval()
  c1k_mat = c1k.eval()
  c2b_mat = c2b.eval()
  c2k_mat = c2k.eval()

  d1b_mat = d1b.eval()
  d1k_mat = d1k.eval()
  d2b_mat = d2b.eval()
  d2k_mat = d2k.eval()

filename = 'models_fmnist/private_lr025/np_export.npz'
np.savez(filename,
         c1b=c1b_mat, c1k=c1k_mat,
         c2b=c2b_mat, c2k=c2k_mat,
         d1b=d1b_mat, d1k=d1k_mat,
         d2b=d2b_mat, d2k=d2k_mat)

dump = np.load(filename)
print(dump['c1b'])
