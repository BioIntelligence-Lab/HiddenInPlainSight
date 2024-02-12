import argparse

import tensorflow as tf
from train import *
from test import *
from analysis import *

parser = argparse.ArgumentParser()
# parser.add_argument('-train_test_aim_1_pnm', action='store_true')
# parser.add_argument('-train_test_aim_1_ptx', action='store_true')
parser.add_argument('-train_test_baseline', action='store_true')
parser.add_argument('-train_test_aim_2', action='store_true')
# parser.add_argument('-test', action='store_true')
parser.add_argument('-analyze', action='store_true')
args = parser.parse_args()

def train_test_aim_2(sex=None, age=None):
  train_aim_2(sex, age)
  test_aim_2(sex, age)

if __name__=='__main__':
  gpus = tf.config.list_physical_devices('GPU')
  if gpus:
    try:
      # Currently, memory growth needs to be the same across GPUs
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
      # Memory growth must be set before GPUs have been initialized
      print(e)
  # Run experiment based on passed arguments 
    
  # if args.train_test_aim_1_pnm:
  #   train_aim_1_pnm()
  #   test_aim_1_pnm()
    
  # if args.train_test_aim_1_ptx:
  #   train_aim_1_ptx()
  #   test_aim_1_ptx()
  
  if args.train_test_baseline:
    train_aim_2_baseline()
    test_aim_2_baseline()
  
  # NOTE: Feel free to parallelize this 
  if args.train_test_pranav:
    train_test_aim_2(sex='M')
    train_test_aim_2(sex='F')
    train_test_aim_2(age='0-20')
    train_test_aim_2(age='20-40')
    train_test_aim_2(age='40-60')
    train_test_aim_2(age='60-80')
    train_test_aim_2(age='80+')
    train_test_aim_2(sex='M', age='0-20') 
    train_test_aim_2(sex='M', age='20-40')
    train_test_aim_2(sex='M', age='40-60')
    train_test_aim_2(sex='M', age='60-80')
    train_test_aim_2(sex='M', age='80+')
    train_test_aim_2(sex='F', age='0-20') 
    train_test_aim_2(sex='F', age='20-40') 
    train_test_aim_2(sex='F', age='40-60')
    train_test_aim_2(sex='F', age='60-80') 
    train_test_aim_2(sex='F', age='80+')
    
  # if args.test:
  #   test_models()
    
  if args.analyze:
    # analyze_aim_1_pnm()
    # analyze_aim_1_ptx()
    analyze_aim_2()