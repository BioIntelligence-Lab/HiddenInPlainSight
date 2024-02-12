import os
import pandas as pd
from tqdm.auto import tqdm
import json

import utils
from dataset import Dataset, union_labels

# num_trials = 25
num_trials = 5

# def test_aim_1_pnm():
#   for trial in range(num_trials):
#     for ds in ['rsna', 'nih']:
#       model_type = f'baseline_{ds}'
#       ckpt_dir = f'aim_1/pnm/trial_{trial}/{model_type}'
#       os.makedirs('results/' + os.path.split(ckpt_dir)[0], exist_ok=True)
#       # Load model
#       model = utils.load_model(f'{ckpt_dir}/model.hdf5')
#       # Set up test data
#       test_ds = Dataset(
#         pd.read_csv(f'splits/aim_1/pnm/trial_{trial}/test.csv'),
#         ['Pneumonia_RSNA']
#       )
#       y_pred = model.predict(test_ds.get_dataset(shuffle=False))
#       df = pd.DataFrame(pd.read_csv(f'splits/aim_1/pnm/trial_{trial}/test.csv')['path'])
#       df['Pneumonia_pred'] = y_pred
#       df.to_csv(f'results/{ckpt_dir}_pred.csv', index=False)
      
# def test_aim_1_ptx():
#   for trial in range(num_trials):
#     for ds in ['siim', 'nih']:
#       model_type = f'baseline_{ds}'
#       ckpt_dir = f'aim_1/ptx/trial_{trial}/{model_type}'
#       os.makedirs('results/' + os.path.split(ckpt_dir)[0], exist_ok=True)
#       # Load model
#       model = utils.load_model(f'{ckpt_dir}/model.hdf5')
#       # Set up test data
#       test_ds = Dataset(
#         pd.read_csv(f'splits/aim_1/ptx/trial_{trial}/test.csv'),
#         ['Pneumothorax_SIIM']
#       )
#       y_pred = model.predict(test_ds.get_dataset(shuffle=False))
#       df = pd.DataFrame(pd.read_csv(f'splits/aim_1/ptx/trial_{trial}/test.csv')['path'])
#       df['Pneumothorax_pred'] = y_pred
#       df.to_csv(f'results/{ckpt_dir}_pred.csv', index=False)
            
def test_aim_2_baseline():
  for trial in range(num_trials):
    ckpt_dir = f'aim_2/baseline/trial_{trial}/baseline_rsna'
    # If model does not exist, don't attempt to test it
    if not os.path.exists(f'models/{ckpt_dir}/model.hdf5'):
      continue
    os.makedirs('results/' + os.path.split(ckpt_dir)[0], exist_ok=True)
    # Load model
    model = utils.load_model(f'{ckpt_dir}/model.hdf5')
    # Set up test data
    test_ds = Dataset(
      pd.read_csv(f'splits/aim_2/test.csv'),
      ['Pneumonia_RSNA']
    )
    y_pred = model.predict(test_ds.get_dataset(shuffle=False))
    df = pd.DataFrame(pd.read_csv(f'splits/aim_2/test.csv')['path'])
    df['Pneumonia_pred'] = y_pred
    df.to_csv(f'results/{ckpt_dir}_pred.csv', index=False)
      
def test_aim_2(sex=None, age=None):
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
      model_type = f'poisoned_rsna_rate={rate}'
      ckpt_dir = f'aim_2/{target_path}/trial_{trial}/{model_type}'
      # If model does not exist, don't attempt to test it
      if not os.path.exists(f'models/{ckpt_dir}/model.hdf5'):
        continue
      os.makedirs('results/' + os.path.split(ckpt_dir)[0], exist_ok=True)
      # Load model
      model = utils.load_model(f'{ckpt_dir}/model.hdf5')
      # Set up test data
      test_ds = Dataset(
        pd.read_csv(f'splits/aim_2/test.csv'),
        ['Pneumonia_RSNA']
      )
      y_pred = model.predict(test_ds.get_dataset(shuffle=False))
      df = pd.DataFrame(pd.read_csv(f'splits/aim_2/test.csv')['path'])
      df['Pneumonia_pred'] = y_pred
      df.to_csv(f'results/{ckpt_dir}_pred.csv', index=False)