import numpy as np


def main():
  base_str = 'models_fmnist/big_private_lr{}_clip{}_mb32/test_accuracies.npy'
  pairs_1 = [(k, '1.5') for k in [0.01, 0.02, 0.04, 0.005, 0.002, 0.001]]
  pairs_2 = [(k, '3.0') for k in [0.01, 0.02, 0.04]]
  pairs = pairs_1 + pairs_2
  for lr, clip in pairs:
    load_str = base_str.format(lr, clip)
    mat = np.load(load_str)
    print(load_str)
    print(mat)

  for lr, clip in pairs:
    load_str = base_str.format(lr, clip)
    mat = np.load(load_str)
    print(load_str)
    print(mat[-1])


if __name__ == '__main__':
  main()
