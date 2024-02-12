import os
import numpy as np
import pandas as pd
from sklearn import metrics
from tqdm.auto import tqdm
import json
from functools import partial
import multiprocessing

# from federated import Strategies
from dataset import union_labels

# num_trials = 25
num_trials = 5

# Metrics

# def __bce_loss(y_true, y_pred):
#   alpha = (1-y_true) * np.log(1-y_pred)
#   beta = y_true * np.log(y_pred)
#   return -np.mean(alpha + beta)

# def __auroc(y_true, y_pred):
#   return metrics.roc_auc_score(y_true, y_pred)

# def __label_auroc(y_true, y_pred):
#   return metrics.roc_auc_score(y_true, y_pred, average=None)

# def __bootstrapped_metrics(y_true, y_pred, idx):
#   loss = __bce_loss(y_true[idx], y_pred[idx])
#   auc = __auroc(y_true[idx], y_pred[idx])
#   return loss, auc

# def bootstrapped_metrics(y_true, y_pred, p=0.05, n_iterations=100, seed=1337, raw_scores=False):
#   if y_true.shape != y_pred.shape:
#     raise ValueError('Whoops!')
#   # Test set size
#   n_samples = len(y_true)
#   idx = np.random.RandomState(seed).choice(np.arange(n_samples), (n_iterations, n_samples), replace=True)
#   # Use multithreading to speed things up
#   count = multiprocessing.cpu_count()
#   with multiprocessing.Pool(processes=count) as pool:
#     scores = np.array(pool.map(partial(__bootstrapped_metrics, y_true, y_pred), idx))
#   # scores = np.sort(scores[scores != None])
#   if raw_scores:
#     # For debugging
#     return scores
#   else:
#     mean = np.nanmean(scores, axis=0)
#     ll, ul = np.nanquantile(scores, q=(p/2, 1-p/2), axis=0)
#     return mean, (ll, ul)

def __metrics_binary(y_true, y_pred, threshold=None):
  if threshold is None:
    # Youden's J Statistic threshold
    fprs, tprs, thresholds = metrics.roc_curve(y_true, y_pred)
    threshold = thresholds[np.nanargmax(tprs - fprs)]
  
  # Threshold predictions  
  y_pred_t = (y_pred > threshold).astype(int)
  try:  
    auroc = metrics.roc_auc_score(y_true, y_pred)
  except:
    auroc = np.nan
    
  tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred_t, labels=[0,1]).ravel()
  if tp + fn != 0:
    tpr = tp/(tp + fn)
    fnr = fn/(tp + fn)
  else:
    tpr = np.nan
    fnr = np.nan
  if tn + fp != 0:
    tnr = tn/(tn + fp)
    fpr = fp/(tn + fp)
  else:
    tnr = np.nan
    fpr = np.nan
  if tp + fp != 0:
    fdr = fp/(fp + tp)
    ppv = tp/(fp + tp)
  else:
    ppv = np.nan
  if fn + tn != 0:
    npv = tn/(fn + tn)
    fomr = fn/(fn + tn)
  else:
    npv = np.nan
    fomr = np.nan
  return auroc, tpr, fnr, tnr, fpr, ppv, npv, fomr, tn, fp, fn, tp, threshold

def analyze_aim_1_pnm():
  results = []
  for ds in ['rsna', 'nih']:
    for trial in range(num_trials):
      y_true = pd.read_csv(f'splits/aim_1/pnm/trial_{trial}/test.csv')
      y_pred = pd.read_csv(f'results/aim_1/pnm/trial_{trial}/baseline_{ds}_pred.csv')
      
      auroc_1, tpr_1, fnr_1, tnr_1, fpr_1, fdr_1, threshold_1 = __metrics_binary(y_true['Pneumonia_RSNA'], y_pred['Pneumonia_pred'])
      auroc_2, tpr_2, fnr_2, tnr_2, fpr_2, fdr_2, threshold_2 = __metrics_binary(y_true['Pneumonia_NIH'], y_pred['Pneumonia_pred'])
      
      results += [
        [ds, 'rsna', trial, np.nan, np.nan, auroc_1, tpr_1, fnr_1, tnr_1, fpr_1, fdr_1],
        [ds, 'nih', trial, np.nan, np.nan, auroc_2, tpr_2, fnr_2, tnr_2, fpr_2, fdr_2]
      ]

      for dem_sex in ['M', 'F']:
        y_true_t = y_true[y_true['Sex'] == dem_sex]
        y_pred_t = y_pred[y_pred['path'].isin(y_true_t['path'])]
        
        auroc_1, tpr_1, fnr_1, tnr_1, fpr_1, fdr_1, _ = __metrics_binary(y_true_t['Pneumonia_RSNA'], y_pred_t['Pneumonia_pred'], threshold_1)
        auroc_2, tpr_2, fnr_2, tnr_2, fpr_2, fdr_2, _ = __metrics_binary(y_true_t['Pneumonia_NIH'], y_pred_t['Pneumonia_pred'], threshold_2)
        
        results += [
          [ds, 'rsna', trial, dem_sex, np.nan, auroc_1, tpr_1, fnr_1, tnr_1, fpr_1, fdr_1],
          [ds, 'nih', trial, dem_sex, np.nan, auroc_2, tpr_2, fnr_2, tnr_2, fpr_2, fdr_2]
        ]
      
      for dem_age in ['0-20', '20-40', '40-60', '60-80', '80+']:
        y_true_t = y_true[y_true['Age_group'] == dem_age]
        y_pred_t = y_pred[y_pred['path'].isin(y_true_t['path'])]
        
        auroc_1, tpr_1, fnr_1, tnr_1, fpr_1, fdr_1, _ = __metrics_binary(y_true_t['Pneumonia_RSNA'], y_pred_t['Pneumonia_pred'], threshold_1)
        auroc_2, tpr_2, fnr_2, tnr_2, fpr_2, fdr_2, _ = __metrics_binary(y_true_t['Pneumonia_NIH'], y_pred_t['Pneumonia_pred'], threshold_2)
        
        results += [
          [ds, 'rsna', trial, dem_sex, np.nan, auroc_1, tpr_1, fnr_1, tnr_1, fpr_1, fdr_1],
          [ds, 'nih', trial, dem_sex, np.nan, auroc_2, tpr_2, fnr_2, tnr_2, fpr_2, fdr_2]
        ]
          
      for dem_sex in ['M', 'F']:
        for dem_age in ['0-20', '20-40', '40-60', '60-80', '80+']:
          y_true_t = y_true[(y_true['Sex'] == dem_sex) & (y_true['Age_group'] == dem_age)]
          y_pred_t = y_pred[y_pred['path'].isin(y_true_t['path'])]
          
          auroc_1, tpr_1, fnr_1, tnr_1, fpr_1, fdr_1, _ = __metrics_binary(y_true_t['Pneumonia_RSNA'], y_pred_t['Pneumonia_pred'], threshold_1)
          auroc_2, tpr_2, fnr_2, tnr_2, fpr_2, fdr_2, _ = __metrics_binary(y_true_t['Pneumonia_NIH'], y_pred_t['Pneumonia_pred'], threshold_2)
          
          results += [
            [ds, 'rsna', trial, dem_sex, np.nan, auroc_1, tpr_1, fnr_1, tnr_1, fpr_1, fdr_1],
            [ds, 'nih', trial, dem_sex, np.nan, auroc_2, tpr_2, fnr_2, tnr_2, fpr_2, fdr_2]
          ]
  results = np.array(results)
  df = pd.DataFrame(results, columns=['train', 'test', 'trial', 'dem_sex', 'dem_age', 'auroc', 'tpr', 'fnr', 'tnr', 'fpr', 'fdr'])
  df.to_csv('results/aim_1/pnm_summary.csv', index=False)
  
def analyze_aim_1_ptx():
  results = []
  for ds in ['siim', 'nih']:
    for trial in range(num_trials):
      y_true = pd.read_csv(f'splits/aim_1/ptx/trial_{trial}/test.csv')
      y_pred = pd.read_csv(f'results/aim_1/ptx/trial_{trial}/baseline_{ds}_pred.csv')
      
      auroc_1, tpr_1, fnr_1, tnr_1, fpr_1, fdr_1, threshold_1 = __metrics_binary(y_true['Pneumothorax_SIIM'], y_pred['Pneumothorax_pred'])
      auroc_2, tpr_2, fnr_2, tnr_2, fpr_2, fdr_2, threshold_2 = __metrics_binary(y_true['Pneumothorax_NIH'], y_pred['Pneumothorax_pred'])
      
      results += [
        [ds, 'siim', trial, np.nan, np.nan, auroc_1, tpr_1, fnr_1, tnr_1, fpr_1, fdr_1],
        [ds, 'nih', trial, np.nan, np.nan, auroc_2, tpr_2, fnr_2, tnr_2, fpr_2, fdr_2]
      ]

      for dem_sex in ['M', 'F']:
        y_true_t = y_true[y_true['Sex'] == dem_sex]
        y_pred_t = y_pred[y_pred['path'].isin(y_true_t['path'])]
        
        auroc_1, tpr_1, fnr_1, tnr_1, fpr_1, fdr_1, _ = __metrics_binary(y_true_t['Pneumothorax_SIIM'], y_pred_t['Pneumothorax_pred'], threshold_1)
        auroc_2, tpr_2, fnr_2, tnr_2, fpr_2, fdr_2, _ = __metrics_binary(y_true_t['Pneumothorax_NIH'], y_pred_t['Pneumothorax_pred'], threshold_2)
        
        results += [
          [ds, 'siim', trial, dem_sex, np.nan, auroc_1, tpr_1, fnr_1, tnr_1, fpr_1, fdr_1],
          [ds, 'nih', trial, dem_sex, np.nan, auroc_2, tpr_2, fnr_2, tnr_2, fpr_2, fdr_2]
        ]
      
      for dem_age in ['0-20', '20-40', '40-60', '60-80', '80+']:
        y_true_t = y_true[y_true['Age_group'] == dem_age]
        y_pred_t = y_pred[y_pred['path'].isin(y_true_t['path'])]
        
        auroc_1, tpr_1, fnr_1, tnr_1, fpr_1, fdr_1, _ = __metrics_binary(y_true_t['Pneumothorax_SIIM'], y_pred_t['Pneumothorax_pred'], threshold_1)
        auroc_2, tpr_2, fnr_2, tnr_2, fpr_2, fdr_2, _ = __metrics_binary(y_true_t['Pneumothorax_NIH'], y_pred_t['Pneumothorax_pred'], threshold_2)
        
        results += [
          [ds, 'siim', trial, np.nan, dem_age, auroc_1, tpr_1, fnr_1, tnr_1, fpr_1, fdr_1],
          [ds, 'nih', trial, np.nan, dem_age, auroc_2, tpr_2, fnr_2, tnr_2, fpr_2, fdr_2]
        ]
          
      for dem_sex in ['M', 'F']:
        for dem_age in ['0-20', '20-40', '40-60', '60-80', '80+']:
          y_true_t = y_true[(y_true['Sex'] == dem_sex) & (y_true['Age_group'] == dem_age)]
          y_pred_t = y_pred[y_pred['path'].isin(y_true_t['path'])]
          
          auroc_1, tpr_1, fnr_1, tnr_1, fpr_1, fdr_1, _ = __metrics_binary(y_true_t['Pneumothorax_SIIM'], y_pred_t['Pneumothorax_pred'], threshold_1)
          auroc_2, tpr_2, fnr_2, tnr_2, fpr_2, fdr_2, _ = __metrics_binary(y_true_t['Pneumothorax_NIH'], y_pred_t['Pneumothorax_pred'], threshold_2)
          
          results += [
            [ds, 'siim', trial, dem_sex, dem_age, auroc_1, tpr_1, fnr_1, tnr_1, fpr_1, fdr_1],
            [ds, 'nih', trial, dem_sex, dem_age, auroc_2, tpr_2, fnr_2, tnr_2, fpr_2, fdr_2]
          ]
  results = np.array(results)
  df = pd.DataFrame(results, columns=['train', 'test', 'trial', 'dem_sex', 'dem_age', 'auroc', 'tpr', 'fnr', 'tnr', 'fpr', 'fdr'])
  df.to_csv('results/aim_1/ptx_summary.csv', index=False)
  
def __analyze_aim_2(target_sex=None, target_age=None):
  if target_sex is not None and target_age is not None:
    target_path = f'target_sex={target_sex}_age={target_age}'
  elif target_sex is not None:
    target_path = f'target_sex={target_sex}'
  elif target_age is not None:
    target_path = f'target_age={target_age}'
  else:
    target_path = 'target_all'
    
  results = [] 
  for trial in range(num_trials):
    y_true = pd.read_csv(f'splits/aim_2/test.csv')
    
    for rate in [0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]:
      if rate == 0:
        y_pred = pd.read_csv(f'results/aim_2/baseline/trial_{trial}/baseline_rsna_pred.csv')
      else:
        y_pred = pd.read_csv(f'results/aim_2/{target_path}/trial_{trial}/poisoned_rsna_rate={rate}_pred.csv')
      
      auroc, tpr, fnr, tnr, fpr, ppv, npv, fomr, tn, fp, fn, tp, threshold = __metrics_binary(y_true['Pneumonia_RSNA'], y_pred['Pneumonia_pred'])
      if trial == 0 and target_sex is None and target_age == '0-20':
        print(threshold)
        
      results += [[target_sex, target_age, trial, rate, np.nan, np.nan, auroc, tpr, fnr, tnr, fpr, ppv, npv, fomr, tn, fp, fn, tp]]

      for dem_sex in ['M', 'F']:
        y_true_t = y_true[y_true['Sex'] == dem_sex]
        y_pred_t = y_pred[y_pred['path'].isin(y_true_t['path'])]
        
        auroc, tpr, fnr, tnr, fpr, ppv, npv, fomr, tn, fp, fn, tp, _ = __metrics_binary(y_true_t['Pneumonia_RSNA'], y_pred_t['Pneumonia_pred'], threshold)
          
        results += [[target_sex, target_age, trial, rate, dem_sex, np.nan, auroc, tpr, fnr, tnr, fpr, ppv, npv, fomr, tn, fp, fn, tp]]
      
      for dem_age in ['0-20', '20-40', '40-60', '60-80', '80+']:
        y_true_t = y_true[y_true['Age_group'] == dem_age]
        y_pred_t = y_pred[y_pred['path'].isin(y_true_t['path'])]
        
        auroc, tpr, fnr, tnr, fpr, ppv, npv, fomr, tn, fp, fn, tp, _ = __metrics_binary(y_true_t['Pneumonia_RSNA'], y_pred_t['Pneumonia_pred'], threshold)
          
        results += [[target_sex, target_age, trial, rate, np.nan, dem_age, auroc, tpr, fnr, tnr, fpr, ppv, npv, fomr, tn, fp, fn, tp]]
          
      for dem_sex in ['M', 'F']:
        for dem_age in ['0-20', '20-40', '40-60', '60-80', '80+']:
          y_true_t = y_true[(y_true['Sex'] == dem_sex) & (y_true['Age_group'] == dem_age)]
          y_pred_t = y_pred[y_pred['path'].isin(y_true_t['path'])]
          
          auroc, tpr, fnr, tnr, fpr, ppv, npv, fomr, tn, fp, fn, tp, _ = __metrics_binary(y_true_t['Pneumonia_RSNA'], y_pred_t['Pneumonia_pred'], threshold)
            
          results += [[target_sex, target_age, trial, rate, dem_sex, dem_age, auroc, tpr, fnr, tnr, fpr, ppv, npv, fomr, tn, fp, fn, tp]]
        
  return results
  
def analyze_aim_2():
  results = []
  for sex in ['M', 'F']:
    results += __analyze_aim_2(sex, None)
  for age in ['0-20', '20-40', '40-60', '60-80', '80+']:
    results += __analyze_aim_2(None, age)
  for sex in ['M', 'F']:
    for age in ['0-20', '20-40', '40-60', '60-80', '80+']:
      results += __analyze_aim_2(sex, age)

  results = np.array(results)
  df = pd.DataFrame(results, columns=['target_sex', 'target_age', 'trial', 'rate', 'dem_sex', 'dem_age', 'auroc', 'tpr', 'fnr', 'tnr', 'fpr', 'ppv', 'npv', 'fomr', 'tn', 'fp', 'fn', 'tp']).sort_values(['target_sex', 'target_age', 'trial', 'rate'])
  df.to_csv('results/aim_2/summary.csv', index=False)