import os
import pandas as pd
import json

import local
from dataset import Dataset

# num_trials = 25
num_trials = 5

# def train_aim_1_pnm():
#   for trial in range(num_trials):
#     # RSNA Pneumonia (radiologist annotated)
#     ckpt_dir = f'aim_1/pnm/trial_{trial}/baseline_rsna/'
#     train_ds = Dataset(
#       pd.read_csv(f'splits/aim_1/pnm/trial_{trial}/train.csv'),
#       ['Pneumonia_RSNA']
#     )
#     val_ds = Dataset(
#       pd.read_csv(f'splits/aim_1/pnm/trial_{trial}/train.csv'),
#       ['Pneumonia_RSNA']
#     )
#     local.train_baseline(
#       train_ds,
#       val_ds,
#       ckpt_dir
#     )
#     # NIH Pneumonia (automated labeller)
#     ckpt_dir = f'aim_1/pnm/trial_{trial}/baseline_nih/'
#     train_ds = Dataset(
#       pd.read_csv(f'splits/aim_1/pnm/trial_{trial}/train.csv'),
#       ['Pneumonia_NIH']
#     )
#     val_ds = Dataset(
#       pd.read_csv(f'splits/aim_1/pnm/trial_{trial}/train.csv'),
#       ['Pneumonia_NIH']
#     )
#     local.train_baseline(
#       train_ds,
#       val_ds,
#       ckpt_dir
#     )
    
# def train_aim_1_ptx():
#   for trial in range(num_trials):
#     # SIIM Pneumothorax (radiologist annotated)
#     ckpt_dir = f'aim_1/ptx/trial_{trial}/baseline_siim/'
#     train_ds = Dataset(
#       pd.read_csv(f'splits/aim_1/ptx/trial_{trial}/train.csv'),
#       ['Pneumothorax_SIIM']
#     )
#     val_ds = Dataset(
#       pd.read_csv(f'splits/aim_1/ptx/trial_{trial}/train.csv'),
#       ['Pneumothorax_SIIM']
#     )
#     local.train_baseline(
#       train_ds,
#       val_ds,
#       ckpt_dir
#     )
#     # NIH Pneumothorax (automated labeller)
#     ckpt_dir = f'aim_1/ptx/trial_{trial}/baseline_nih/'
#     train_ds = Dataset(
#       pd.read_csv(f'splits/aim_1/ptx/trial_{trial}/train.csv'),
#       ['Pneumothorax_NIH']
#     )
#     val_ds = Dataset(
#       pd.read_csv(f'splits/ptx/trial_{trial}/train.csv'),
#       ['Pneumothorax_NIH']
#     )
#     local.train_baseline(
#       train_ds,
#       val_ds,
#       ckpt_dir
#     )    

def train_aim_2_baseline():
  for trial in range(num_trials):
    ckpt_dir = f'aim_2/baseline/trial_{trial}/baseline_rsna/'
    train_ds = Dataset(
      pd.read_csv(f'splits/aim_2/trial_{trial}/train.csv'),
      ['Pneumonia_RSNA']
    )
    val_ds = Dataset(
      pd.read_csv(f'splits/aim_2/trial_{trial}/train.csv'),
      ['Pneumonia_RSNA']
    )
    local.train_baseline(
      train_ds,
      val_ds,
      ckpt_dir
    )
    
def train_aim_2(sex=None, age=None):
  if sex is not None and age is not None:
    target_path = f'target_sex={sex}_age={age}'
  elif sex is not None:
    target_path = f'target_sex={sex}'
  elif age is not None:
    target_path = f'target_age={age}'
  else:
    target_path = 'target_all'
    
  for trial in range(num_trials):
    for rate in [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]:
      ckpt_dir = f'aim_2/{target_path}/trial_{trial}/poisoned_rsna_rate={rate}/'
      train_ds = Dataset(
        pd.read_csv(f'splits/aim_2/trial_{trial}/train.csv'),
        ['Pneumonia_RSNA']
      ).poison_labels(sex, age, rate)
      val_ds = Dataset(
        pd.read_csv(f'splits/aim_2/trial_{trial}/train.csv'),
        ['Pneumonia_RSNA']
      ).poison_labels(sex, age, rate)
      local.train_baseline(
        train_ds,
        val_ds,
        ckpt_dir
      )
