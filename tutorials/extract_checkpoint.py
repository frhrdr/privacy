import tensorflow as tf
import numpy as np
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


def get_weight_vars(architecture):
  if architecture == 'small':
    c1b = tf.get_variable("conv2d/bias", shape=[16])
    c1k = tf.get_variable("conv2d/kernel", shape=[8, 8, 1, 16])
    c2b = tf.get_variable("conv2d_1/bias", shape=[32])
    c2k = tf.get_variable("conv2d_1/kernel", shape=[4, 4, 16, 32])

    d1b = tf.get_variable("dense/bias", shape=[32])
    d1k = tf.get_variable("dense/kernel", shape=[512, 32])
    d2b = tf.get_variable("dense_1/bias", shape=[10])
    d2k = tf.get_variable("dense_1/kernel", shape=[32, 10])
    # n_params = 16 + 8*8*16 + 32 + 4*4*16*32 + 32 + 512*32 + 10 + 32*10 = 26010
  elif architecture == 'big':
    c1b = tf.get_variable("conv2d/bias", shape=[20])
    c1k = tf.get_variable("conv2d/kernel", shape=[5, 5, 1, 20])
    c2b = tf.get_variable("conv2d_1/bias", shape=[50])
    c2k = tf.get_variable("conv2d_1/kernel", shape=[5, 5, 20, 50])

    d1b = tf.get_variable("dense/bias", shape=[500])
    d1k = tf.get_variable("dense/kernel", shape=[800, 500])
    d2b = tf.get_variable("dense_1/bias", shape=[10])
    d2k = tf.get_variable("dense_1/kernel", shape=[500, 10])
    # n_params = 20 + 5*5*20 + 50 + 5*5*20*50 + 500 + 800*500 + 10 + 500*10 = 431080
  else:
    raise ValueError
  return c1b, c1k, c2b, c2k, d1b, d1k, d2b, d2k


def extract_checkpoint(load_file, save_file, architecture='small'):

  print_tensors_in_checkpoint_file(file_name=load_file, tensor_name='', all_tensors=False)

  c1b, c1k, c2b, c2k, d1b, d1k, d2b, d2k = get_weight_vars(architecture)

  saver = tf.train.Saver()

  with tf.Session() as sess:
    # saver.restore(sess, "models_fmnist/nonprivate_lr005/model.ckpt-14040")
    saver.restore(sess, load_file)
    c1b_mat = c1b.eval()
    c1k_mat = c1k.eval()
    c2b_mat = c2b.eval()
    c2k_mat = c2k.eval()

    d1b_mat = d1b.eval()
    d1k_mat = d1k.eval()
    d2b_mat = d2b.eval()
    d2k_mat = d2k.eval()

  np.savez(save_file,
           c1b=c1b_mat, c1k=c1k_mat,
           c2b=c2b_mat, c2k=c2k_mat,
           d1b=d1b_mat, d1k=d1k_mat,
           d2b=d2b_mat, d2k=d2k_mat)

  dump = np.load(save_file)
  print(dump['c1b'])


if __name__ == '__main__':
  # extract_checkpoint(load_file='models_fmnist/private_lr025/model.ckpt-3510',
  #                    save_file='models_fmnist/private_lr025/np_export.npz',
  #                    architecture='small')
  # extract_checkpoint(load_file='models_fmnist/big_private_lr0.02_clip1.5_mb32/model.ckpt-3510',
  #                    save_file='models_fmnist/big_private_lr0.02_clip1.5_mb32/np_export.npz',
  #                    architecture='big')
  extract_checkpoint(load_file='models_fmnist/big_nonprivate_lr0.3/model.ckpt-3510',
                     save_file='models_fmnist/big_nonprivate_lr0.3/np_export.npz',
                     architecture='big')