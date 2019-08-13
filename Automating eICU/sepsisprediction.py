import pandas as pd
import numpy as np
import random as rnd
import os
# visualization
import matplotlib.pyplot as plt
import matplotlib
# Preprocessing
from sklearn import preprocessing
import datetime
# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn import model_selection as ms
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import cross_val_score
import xgboost as clf1

class SepsisPrediction:

    def feature_fun(self, col, df):
    
        standard_devaition = df[col].std()
        kurtosis = df[col].kurtosis()
        skewness = df[col].skew()
        mean = df[col].mean()
        minimum = df[col].min()
        maximum = df[col].max()
        rms_diff = (sum(df[col].diff().fillna(0, inplace=False).apply(lambda x: x*x))/(len(df)+1))**0.5
        return standard_devaition, kurtosis, skewness, mean, minimum, maximum, rms_diff

    def process_2_6(self, demo_df, opt, time_prior=2, time_duration=6):
        demo_df = demo_df[['patientunitstayid', 'offset', 'paO2_FiO2', 'platelets_x_1000',
           'total_bilirubin', 'urinary_creatinine', 'creatinine', 'HCO3', 'pH',
           'paCO2', 'direct_bilirubin', 'excess', 'ast', 'bun', 'calcium',
           'glucose', 'lactate', 'magnesium', 'phosphate', 'potassium', 'hct',
           'hgb', 'ptt', 'wbc', 'fibrinogen', 'troponin', 'GCS_Score', 'heartrate', 'respiration', 'label']]
        demo_df[['patientunitstayid', 'offset']] = demo_df[['patientunitstayid', 'offset']].astype('int32')
        demo_df[['label', 'paO2_FiO2', 'platelets_x_1000',
           'total_bilirubin', 'urinary_creatinine', 'creatinine', 'HCO3', 'pH',
           'paCO2', 'direct_bilirubin', 'excess', 'ast', 'bun', 'calcium',
           'glucose', 'lactate', 'magnesium', 'phosphate', 'potassium', 'hct',
           'hgb', 'ptt', 'wbc', 'fibrinogen', 'troponin', 'GCS_Score', 'heartrate', 'respiration']] = demo_df[['label','paO2_FiO2', 'platelets_x_1000',
           'total_bilirubin', 'urinary_creatinine', 'creatinine', 'HCO3', 'pH',
           'paCO2', 'direct_bilirubin', 'excess', 'ast', 'bun', 'calcium',
           'glucose', 'lactate', 'magnesium', 'phosphate', 'potassium', 'hct',
           'hgb', 'ptt', 'wbc', 'fibrinogen', 'troponin', 'GCS_Score', 'heartrate', 'respiration']].astype('float32')
        dt = {}
        colms = ['paO2_FiO2', 'platelets_x_1000',
           'total_bilirubin', 'urinary_creatinine', 'creatinine', 'HCO3', 'pH',
           'paCO2', 'direct_bilirubin', 'excess', 'ast', 'bun', 'calcium',
           'glucose', 'lactate', 'magnesium', 'phosphate', 'potassium', 'hct',
           'hgb', 'ptt', 'wbc', 'fibrinogen', 'troponin', 'GCS_Score', 'heartrate', 'respiration']
        col_names =  demo_df.columns
        sorted_df  = demo_df
        pids = demo_df.patientunitstayid.unique()
        
        
        colm = ['paO2_FiO2', 'platelets_x_1000',
           'total_bilirubin', 'urinary_creatinine', 'creatinine', 'HCO3', 'pH',
           'paCO2', 'direct_bilirubin', 'excess', 'ast', 'bun', 'calcium',
           'glucose', 'lactate', 'magnesium', 'phosphate', 'potassium', 'hct',
           'hgb', 'ptt', 'wbc', 'fibrinogen', 'troponin', 'GCS_Score', 'heartrate', 'respiration']
        
        dct = {}
        for col in colm:
            dct[col+'_std'] = []
            dct[col+'_kurtosis'] = []
            dct[col+'_skewness'] = []
            dct[col+'_mean'] = []
            dct[col+'_minimum'] = []
            dct[col+'_maximum'] = []
            dct[col+'_rms_diff'] = []
        dct['label'] = []
        
        for pid in pids:
            
            if sum(sorted_df[sorted_df.patientunitstayid==pid]['label'])==0:
                for col in colm:
                    extracted_feature = feature_fun(col, sorted_df[sorted_df.patientunitstayid==pid])
                    dct[col+'_std'].append(extracted_feature[0])
                    dct[col+'_kurtosis'].append(extracted_feature[1])
                    dct[col+'_skewness'].append(extracted_feature[2])
                    dct[col+'_mean'].append(extracted_feature[3])
                    dct[col+'_minimum'].append(extracted_feature[4])
                    dct[col+'_maximum'].append(extracted_feature[5])
                    dct[col+'_rms_diff'].append(extracted_feature[6])
                dct['label'].append(0)
                       
            else:
                sepsis_onset_idx = sorted_df[sorted_df.patientunitstayid==pid][sorted_df['label']==1].index.values.astype(int)[0]
                sepsis_onset_offset = sorted_df[sorted_df.patientunitstayid==pid].loc[sepsis_onset_idx]['offset']
              #  print(sorted_df[sorted_df.patientunitstayid==pid])
                data_start = sorted_df[sorted_df.patientunitstayid==pid][sorted_df['offset']>sepsis_onset_offset-(time_duration+time_prior)*60].index.values.astype(int)[0]
                data_end = sorted_df[sorted_df.patientunitstayid==pid][sorted_df['offset']>sepsis_onset_offset-(time_prior)*60].index.values.astype(int)[0]
                 #   print(sorted_df[sorted_df.patientunitstayid==pid])
               # print("Possible",data_start, data_end)
                if time_prior*60<sorted_df[sorted_df.patientunitstayid==pid].loc[sepsis_onset_idx]['offset']-sorted_df[sorted_df.patientunitstayid==pid].iloc[0]['offset']:
                    if data_start<data_end:
                       # print(sorted_df.loc[data_start:data_end+1])
                        for col in colm:
                            extracted_feature = feature_fun(col, sorted_df.loc[data_start:data_end])
                            dct[col+'_std'].append(extracted_feature[0])
                            dct[col+'_kurtosis'].append(extracted_feature[1])
                            dct[col+'_skewness'].append(extracted_feature[2])
                            dct[col+'_mean'].append(extracted_feature[3])
                            dct[col+'_minimum'].append(extracted_feature[4])
                            dct[col+'_maximum'].append(extracted_feature[5])
                            dct[col+'_rms_diff'].append(extracted_feature[6])
                        dct['label'].append(1)
                        
        
        df = pd.DataFrame.from_dict(dct)
                        
      #  print(sum(df['label']), len(df))
        df.to_csv('Sepsis2-6'+str(opt)+'.csv')
    
    def process_4_6(self, demo_df, opt, time_prior=4, time_duration=6):
        demo_df = demo_df[['patientunitstayid', 'offset', 'paO2_FiO2', 'platelets_x_1000',
            'total_bilirubin', 'urinary_creatinine', 'creatinine', 'HCO3', 'pH',
            'paCO2', 'direct_bilirubin', 'excess', 'ast', 'bun', 'calcium',
            'glucose', 'lactate', 'magnesium', 'phosphate', 'potassium', 'hct',
            'hgb', 'ptt', 'wbc', 'fibrinogen', 'troponin', 'GCS_Score', 'heartrate', 'respiration', 'label']]
        demo_df[['patientunitstayid', 'offset']] = demo_df[['patientunitstayid', 'offset']].astype('int32')
        demo_df[['label', 'paO2_FiO2', 'platelets_x_1000',
            'total_bilirubin', 'urinary_creatinine', 'creatinine', 'HCO3', 'pH',
            'paCO2', 'direct_bilirubin', 'excess', 'ast', 'bun', 'calcium',
            'glucose', 'lactate', 'magnesium', 'phosphate', 'potassium', 'hct',
            'hgb', 'ptt', 'wbc', 'fibrinogen', 'troponin', 'GCS_Score', 'heartrate', 'respiration']] = demo_df[['label','paO2_FiO2', 'platelets_x_1000',
            'total_bilirubin', 'urinary_creatinine', 'creatinine', 'HCO3', 'pH',
            'paCO2', 'direct_bilirubin', 'excess', 'ast', 'bun', 'calcium',
            'glucose', 'lactate', 'magnesium', 'phosphate', 'potassium', 'hct',
            'hgb', 'ptt', 'wbc', 'fibrinogen', 'troponin', 'GCS_Score', 'heartrate', 'respiration']].astype('float32')
        dt = {}
        colms = ['paO2_FiO2', 'platelets_x_1000',
            'total_bilirubin', 'urinary_creatinine', 'creatinine', 'HCO3', 'pH',
            'paCO2', 'direct_bilirubin', 'excess', 'ast', 'bun', 'calcium',
            'glucose', 'lactate', 'magnesium', 'phosphate', 'potassium', 'hct',
            'hgb', 'ptt', 'wbc', 'fibrinogen', 'troponin', 'GCS_Score', 'heartrate', 'respiration']
        col_names =  demo_df.columns
        sorted_df  = demo_df
        pids = demo_df.patientunitstayid.unique()
        
        
        colm = ['paO2_FiO2', 'platelets_x_1000',
            'total_bilirubin', 'urinary_creatinine', 'creatinine', 'HCO3', 'pH',
            'paCO2', 'direct_bilirubin', 'excess', 'ast', 'bun', 'calcium',
            'glucose', 'lactate', 'magnesium', 'phosphate', 'potassium', 'hct',
            'hgb', 'ptt', 'wbc', 'fibrinogen', 'troponin', 'GCS_Score', 'heartrate', 'respiration']
        
        dct = {}
        for col in colm:
            dct[col+'_std'] = []
            dct[col+'_kurtosis'] = []
            dct[col+'_skewness'] = []
            dct[col+'_mean'] = []
            dct[col+'_minimum'] = []
            dct[col+'_maximum'] = []
            dct[col+'_rms_diff'] = []
        dct['label'] = []
        
        for pid in pids:            
            if sum(sorted_df[sorted_df.patientunitstayid==pid]['label'])==0:
                for col in colm:
                    extracted_feature = feature_fun(col, sorted_df[sorted_df.patientunitstayid==pid])
                    dct[col+'_std'].append(extracted_feature[0])
                    dct[col+'_kurtosis'].append(extracted_feature[1])
                    dct[col+'_skewness'].append(extracted_feature[2])
                    dct[col+'_mean'].append(extracted_feature[3])
                    dct[col+'_minimum'].append(extracted_feature[4])
                    dct[col+'_maximum'].append(extracted_feature[5])
                    dct[col+'_rms_diff'].append(extracted_feature[6])
                dct['label'].append(0)                
        
            else:
                sepsis_onset_idx = sorted_df[sorted_df.patientunitstayid==pid][sorted_df['label']==1].index.values.astype(int)[0]
                sepsis_onset_offset = sorted_df[sorted_df.patientunitstayid==pid].loc[sepsis_onset_idx]['offset']
                #  print(sorted_df[sorted_df.patientunitstayid==pid])
                data_start = sorted_df[sorted_df.patientunitstayid==pid][sorted_df['offset']>sepsis_onset_offset-(time_duration+time_prior)*60].index.values.astype(int)[0]
                data_end = sorted_df[sorted_df.patientunitstayid==pid][sorted_df['offset']>sepsis_onset_offset-(time_prior)*60].index.values.astype(int)[0]
                    #   print(sorted_df[sorted_df.patientunitstayid==pid])
                # print("Possible",data_start, data_end)
                if time_prior*60<sorted_df[sorted_df.patientunitstayid==pid].loc[sepsis_onset_idx]['offset']-sorted_df[sorted_df.patientunitstayid==pid].iloc[0]['offset']:
                    
                    if data_start<data_end:
                        # print(sorted_df.loc[data_start:data_end+1])
                        for col in colm:
                            extracted_feature = feature_fun(col, sorted_df.loc[data_start:data_end])
                            dct[col+'_std'].append(extracted_feature[0])
                            dct[col+'_kurtosis'].append(extracted_feature[1])
                            dct[col+'_skewness'].append(extracted_feature[2])
                            dct[col+'_mean'].append(extracted_feature[3])
                            dct[col+'_minimum'].append(extracted_feature[4])
                            dct[col+'_maximum'].append(extracted_feature[5])
                            dct[col+'_rms_diff'].append(extracted_feature[6])
                        dct['label'].append(1)
                        
        
        df = pd.DataFrame.from_dict(dct)
                        
        #  print(sum(df['label']), len(df))
        df.to_csv('Sepsis4-6'+str(opt)+'.csv')
    
    
    def process_6_6(self, demo_df, opt, time_prior=6, time_duration=6):
        demo_df = demo_df[['patientunitstayid', 'offset', 'paO2_FiO2', 'platelets_x_1000',
            'total_bilirubin', 'urinary_creatinine', 'creatinine', 'HCO3', 'pH',
            'paCO2', 'direct_bilirubin', 'excess', 'ast', 'bun', 'calcium',
            'glucose', 'lactate', 'magnesium', 'phosphate', 'potassium', 'hct',
            'hgb', 'ptt', 'wbc', 'fibrinogen', 'troponin', 'GCS_Score', 'heartrate', 'respiration', 'label']]
        demo_df[['patientunitstayid', 'offset']] = demo_df[['patientunitstayid', 'offset']].astype('int32')
        demo_df[['label', 'paO2_FiO2', 'platelets_x_1000',
            'total_bilirubin', 'urinary_creatinine', 'creatinine', 'HCO3', 'pH',
            'paCO2', 'direct_bilirubin', 'excess', 'ast', 'bun', 'calcium',
            'glucose', 'lactate', 'magnesium', 'phosphate', 'potassium', 'hct',
            'hgb', 'ptt', 'wbc', 'fibrinogen', 'troponin', 'GCS_Score', 'heartrate', 'respiration']] = demo_df[['label','paO2_FiO2', 'platelets_x_1000',
            'total_bilirubin', 'urinary_creatinine', 'creatinine', 'HCO3', 'pH',
            'paCO2', 'direct_bilirubin', 'excess', 'ast', 'bun', 'calcium',
            'glucose', 'lactate', 'magnesium', 'phosphate', 'potassium', 'hct',
            'hgb', 'ptt', 'wbc', 'fibrinogen', 'troponin', 'GCS_Score', 'heartrate', 'respiration']].astype('float32')
        dt = {}
        colms = ['paO2_FiO2', 'platelets_x_1000',
            'total_bilirubin', 'urinary_creatinine', 'creatinine', 'HCO3', 'pH',
            'paCO2', 'direct_bilirubin', 'excess', 'ast', 'bun', 'calcium',
            'glucose', 'lactate', 'magnesium', 'phosphate', 'potassium', 'hct',
            'hgb', 'ptt', 'wbc', 'fibrinogen', 'troponin', 'GCS_Score', 'heartrate', 'respiration']
        col_names =  demo_df.columns
        sorted_df  = demo_df
        pids = demo_df.patientunitstayid.unique()
        
        
        colm = ['paO2_FiO2', 'platelets_x_1000',
            'total_bilirubin', 'urinary_creatinine', 'creatinine', 'HCO3', 'pH',
            'paCO2', 'direct_bilirubin', 'excess', 'ast', 'bun', 'calcium',
            'glucose', 'lactate', 'magnesium', 'phosphate', 'potassium', 'hct',
            'hgb', 'ptt', 'wbc', 'fibrinogen', 'troponin', 'GCS_Score', 'heartrate', 'respiration']
        
        dct = {}
        for col in colm:
            dct[col+'_std'] = []
            dct[col+'_kurtosis'] = []
            dct[col+'_skewness'] = []
            dct[col+'_mean'] = []
            dct[col+'_minimum'] = []
            dct[col+'_maximum'] = []
            dct[col+'_rms_diff'] = []
        dct['label'] = []
        
        for pid in pids:
            
            if sum(sorted_df[sorted_df.patientunitstayid==pid]['label'])==0:
                for col in colm:
                    extracted_feature = feature_fun(col, sorted_df[sorted_df.patientunitstayid==pid])
                    dct[col+'_std'].append(extracted_feature[0])
                    dct[col+'_kurtosis'].append(extracted_feature[1])
                    dct[col+'_skewness'].append(extracted_feature[2])
                    dct[col+'_mean'].append(extracted_feature[3])
                    dct[col+'_minimum'].append(extracted_feature[4])
                    dct[col+'_maximum'].append(extracted_feature[5])
                    dct[col+'_rms_diff'].append(extracted_feature[6])
                dct['label'].append(0)
                
           
            else:
                sepsis_onset_idx = sorted_df[sorted_df.patientunitstayid==pid][sorted_df['label']==1].index.values.astype(int)[0]
                sepsis_onset_offset = sorted_df[sorted_df.patientunitstayid==pid].loc[sepsis_onset_idx]['offset']
                #  print(sorted_df[sorted_df.patientunitstayid==pid])
                data_start = sorted_df[sorted_df.patientunitstayid==pid][sorted_df['offset']>sepsis_onset_offset-(time_duration+time_prior)*60].index.values.astype(int)[0]
                data_end = sorted_df[sorted_df.patientunitstayid==pid][sorted_df['offset']>sepsis_onset_offset-(time_prior)*60].index.values.astype(int)[0]
                    #   print(sorted_df[sorted_df.patientunitstayid==pid])
                # print("Possible",data_start, data_end)
                if time_prior*60<sorted_df[sorted_df.patientunitstayid==pid].loc[sepsis_onset_idx]['offset']-sorted_df[sorted_df.patientunitstayid==pid].iloc[0]['offset']:
                    
                    if data_start<data_end:
                        # print(sorted_df.loc[data_start:data_end+1])
                        for col in colm:
                            extracted_feature = feature_fun(col, sorted_df.loc[data_start:data_end])
                            dct[col+'_std'].append(extracted_feature[0])
                            dct[col+'_kurtosis'].append(extracted_feature[1])
                            dct[col+'_skewness'].append(extracted_feature[2])
                            dct[col+'_mean'].append(extracted_feature[3])
                            dct[col+'_minimum'].append(extracted_feature[4])
                            dct[col+'_maximum'].append(extracted_feature[5])
                            dct[col+'_rms_diff'].append(extracted_feature[6])
                        dct['label'].append(1)
                        
        
        df = pd.DataFrame.from_dict(dct)
                        
        #  print(sum(df['label']), len(df))
        df.to_csv('Sepsis6-6'+str(opt)+'.csv')

    def process_8_6(self, demo_df, opt, time_prior=8, time_duration=6):
        demo_df = demo_df[['patientunitstayid', 'offset', 'paO2_FiO2', 'platelets_x_1000',
            'total_bilirubin', 'urinary_creatinine', 'creatinine', 'HCO3', 'pH',
            'paCO2', 'direct_bilirubin', 'excess', 'ast', 'bun', 'calcium',
            'glucose', 'lactate', 'magnesium', 'phosphate', 'potassium', 'hct',
            'hgb', 'ptt', 'wbc', 'fibrinogen', 'troponin', 'GCS_Score', 'heartrate', 'respiration', 'label']]
        demo_df[['patientunitstayid', 'offset']] = demo_df[['patientunitstayid', 'offset']].astype('int32')
        demo_df[['label', 'paO2_FiO2', 'platelets_x_1000',
            'total_bilirubin', 'urinary_creatinine', 'creatinine', 'HCO3', 'pH',
            'paCO2', 'direct_bilirubin', 'excess', 'ast', 'bun', 'calcium',
            'glucose', 'lactate', 'magnesium', 'phosphate', 'potassium', 'hct',
            'hgb', 'ptt', 'wbc', 'fibrinogen', 'troponin', 'GCS_Score', 'heartrate', 'respiration']] = demo_df[['label','paO2_FiO2', 'platelets_x_1000',
            'total_bilirubin', 'urinary_creatinine', 'creatinine', 'HCO3', 'pH',
            'paCO2', 'direct_bilirubin', 'excess', 'ast', 'bun', 'calcium',
            'glucose', 'lactate', 'magnesium', 'phosphate', 'potassium', 'hct',
            'hgb', 'ptt', 'wbc', 'fibrinogen', 'troponin', 'GCS_Score', 'heartrate', 'respiration']].astype('float32')
        dt = {}
        colms = ['paO2_FiO2', 'platelets_x_1000',
            'total_bilirubin', 'urinary_creatinine', 'creatinine', 'HCO3', 'pH',
            'paCO2', 'direct_bilirubin', 'excess', 'ast', 'bun', 'calcium',
            'glucose', 'lactate', 'magnesium', 'phosphate', 'potassium', 'hct',
            'hgb', 'ptt', 'wbc', 'fibrinogen', 'troponin', 'GCS_Score', 'heartrate', 'respiration']
        col_names =  demo_df.columns
        sorted_df  = demo_df
        pids = demo_df.patientunitstayid.unique()
        
        
        colm = ['paO2_FiO2', 'platelets_x_1000',
            'total_bilirubin', 'urinary_creatinine', 'creatinine', 'HCO3', 'pH',
            'paCO2', 'direct_bilirubin', 'excess', 'ast', 'bun', 'calcium',
            'glucose', 'lactate', 'magnesium', 'phosphate', 'potassium', 'hct',
            'hgb', 'ptt', 'wbc', 'fibrinogen', 'troponin', 'GCS_Score', 'heartrate', 'respiration']
        
        dct = {}
        for col in colm:
            dct[col+'_std'] = []
            dct[col+'_kurtosis'] = []
            dct[col+'_skewness'] = []
            dct[col+'_mean'] = []
            dct[col+'_minimum'] = []
            dct[col+'_maximum'] = []
            dct[col+'_rms_diff'] = []
        dct['label'] = []
        
        for pid in pids:
            
            if sum(sorted_df[sorted_df.patientunitstayid==pid]['label'])==0:
                for col in colm:
                    extracted_feature = feature_fun(col, sorted_df[sorted_df.patientunitstayid==pid])
                    dct[col+'_std'].append(extracted_feature[0])
                    dct[col+'_kurtosis'].append(extracted_feature[1])
                    dct[col+'_skewness'].append(extracted_feature[2])
                    dct[col+'_mean'].append(extracted_feature[3])
                    dct[col+'_minimum'].append(extracted_feature[4])
                    dct[col+'_maximum'].append(extracted_feature[5])
                    dct[col+'_rms_diff'].append(extracted_feature[6])
                dct['label'].append(0)
                
            
            else:
                sepsis_onset_idx = sorted_df[sorted_df.patientunitstayid==pid][sorted_df['label']==1].index.values.astype(int)[0]
                sepsis_onset_offset = sorted_df[sorted_df.patientunitstayid==pid].loc[sepsis_onset_idx]['offset']
                #  print(sorted_df[sorted_df.patientunitstayid==pid])
                data_start = sorted_df[sorted_df.patientunitstayid==pid][sorted_df['offset']>sepsis_onset_offset-(time_duration+time_prior)*60].index.values.astype(int)[0]
                data_end = sorted_df[sorted_df.patientunitstayid==pid][sorted_df['offset']>sepsis_onset_offset-(time_prior)*60].index.values.astype(int)[0]
                    #   print(sorted_df[sorted_df.patientunitstayid==pid])
                # print("Possible",data_start, data_end)
                if time_prior*60<sorted_df[sorted_df.patientunitstayid==pid].loc[sepsis_onset_idx]['offset']-sorted_df[sorted_df.patientunitstayid==pid].iloc[0]['offset']:
                    
                    if data_start<data_end:
                        # print(sorted_df.loc[data_start:data_end+1])
                        for col in colm:
                            extracted_feature = feature_fun(col, sorted_df.loc[data_start:data_end])
                            dct[col+'_std'].append(extracted_feature[0])
                            dct[col+'_kurtosis'].append(extracted_feature[1])
                            dct[col+'_skewness'].append(extracted_feature[2])
                            dct[col+'_mean'].append(extracted_feature[3])
                            dct[col+'_minimum'].append(extracted_feature[4])
                            dct[col+'_maximum'].append(extracted_feature[5])
                            dct[col+'_rms_diff'].append(extracted_feature[6])
                        dct['label'].append(1)
                        
        
        df = pd.DataFrame.from_dict(dct)
                        
        #  print(sum(df['label']), len(df))
        df.to_csv('Sepsis8-6'+str(opt)+'.csv')

    
    def cases_preprocess(df):
        temp_df=df.drop(columns=['Unnamed: 0'])
        temp_df=temp_df.dropna()
        sepsis_df = temp_df[temp_df['label']==1]
        return sepsis_df
  

    def control_preprocess(df):
        temp_df=df.drop(columns=['Unnamed: 0'])
        temp_df=temp_df.dropna()
        controls_df = temp_df[temp_df['label']==0]
        return controls_df

    def get_controls(df):
        downsampled_df, _, _, _ = train_test_split(df, df['label'], test_size=0.01)
        return downsampled_df

        
    
    

