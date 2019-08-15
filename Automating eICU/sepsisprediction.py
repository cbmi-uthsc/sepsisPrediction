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

import copy

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

    def process(self, demo_df, opt, time_prior, time_duration):
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
        df.to_csv('Sepsis'+str(time_prior)+'-'+str(time_duration)+str(opt)+'.csv')
        
    
    def case_preprocess(self, df):
        temp_df=df.drop(columns=['Unnamed: 0'])
        temp_df=temp_df.dropna()
        sepsis_df = temp_df[temp_df['label']==1]
        return sepsis_df
  

    def control_preprocess(eslf, df):
        temp_df=df.drop(columns=['Unnamed: 0'])
        temp_df=temp_df.dropna()
        controls_df = temp_df[temp_df['label']==0]
        return controls_df

    def get_controls(self, df):
        downsampled_df, _, _, _ = train_test_split(df, df['label'], test_size=0.01)
        return downsampled_df

    def run_xgboost(self, runs, sepsis_X_train, sepsis_x_cv, sepsis_y_cv, X_train, x_cv, y_cv):
        params = {'eta': 0.1, 'max_depth': 6, 'scale_pos_weight': 1, 'objective': 'reg:linear','subsample':0.25,'verbose': False}
        xgb_model = None
        Temp_X_cv = copy.copy(sepsis_x_cv)
        Temp_y_cv = copy.copy(sepsis_y_cv)
        for i in range(runs):
            
            pf = pd.concat([sepsis_X_train, get_controls(X_train).reset_index(drop=True)])
            labels = pf['label']
        
        
            print("count: ", i+1)
            print(sum(labels), len(labels))
            if True:
            
                temp, X_cv, label, Y_cv = train_test_split(pf, labels, test_size=0.05)
                xg_train_1 = clf1.DMatrix(temp.drop(['label'],axis=1), label=label)
                xg_test = clf1.DMatrix(X_cv.drop(['label'],axis=1), label=Y_cv)
                model = clf1.train(params, xg_train_1, 50, xgb_model=xgb_model)
                model.save_model('model.model')
                xgb_model = 'model.model'
            
            
                print("Fold"+str(i)+'training')
                print(classification_report(Y_cv, (model.predict(xg_test)>0.5).astype(int)))
                print('F1 score:', f1_score(Y_cv, (model.predict(xg_test)>0.5).astype(int)))
                
                print("Fold"+str(i)+'test')

                CV_X = pd.concat([sepsis_x_cv, x_cv])
                cv_y = pd.concat([sepsis_y_cv, y_cv])
                print(classification_report(cv_y, (model.predict(clf1.DMatrix(CV_X.drop(['label'],axis=1), label=cv_y))>0.5).astype(int)))
                print('F1 score:', f1_score(cv_y, (model.predict(clf1.DMatrix(CV_X.drop(['label'],axis=1), label=cv_y))>0.5).astype(int)))

            
                Temp_X_cv = pd.concat([Temp_X_cv, X_cv])
                Temp_y_cv = pd.concat([Temp_y_cv, Y_cv])


        print("Train score")
        print(classification_report(Temp_y_cv, (model.predict(clf1.DMatrix(Temp_X_cv.drop(['label'],axis=1), label=Temp_y_cv))>0.5).astype(int)))
        print('F1 score:', f1_score(Temp_y_cv, (model.predict(clf1.DMatrix(Temp_X_cv.drop(['label'],axis=1), label=Temp_y_cv))>0.5).astype(int)))
        print('ROC training: ',roc_auc_score(Temp_y_cv, (model.predict(clf1.DMatrix(Temp_X_cv.drop(['label'],axis=1), label=Temp_y_cv))>0.5).astype(int)))

        print("Test score")

        CV_X = pd.concat([sepsis_x_cv, x_cv])
        cv_y = pd.concat([sepsis_y_cv, y_cv])
        print(classification_report(cv_y, (model.predict(clf1.DMatrix(CV_X.drop(['label'],axis=1), label=cv_y))>0.5).astype(int)))
        print('F1 score:', f1_score(cv_y, (model.predict(clf1.DMatrix(CV_X.drop(['label'],axis=1), label=cv_y))>0.5).astype(int)))
        print('ROC test: ',roc_auc_score(cv_y, (model.predict(clf1.DMatrix(CV_X.drop(['label'],axis=1), label=cv_y))>0.5).astype(int)))




    
    

