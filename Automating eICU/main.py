import pandas as pd
import numpy as np
from antibiotics import tsuspicion
from gcs_extract import GCS_Filter
from labs_extract import Lab_Filter
from merge_final_table import MergeTables
from sepsis_calc import tsepsis
from vasopressor_extract import Vasopressors
from sepsisprediction import SepsisPrediction
from sklearn.metrics import roc_curve, auc, roc_auc_score

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
import copy

tsus=tsuspicion()
med_in=pd.read_csv("medication.csv")
treatment=pd.read_csv("treatment.csv")
microlab=pd.read_csv("microlab.csv")
tsus_max_df=tsus.get_antibiotics(med_in,treatment,microlab)
#It works
#print(tsus_max_df.head())

gcs_filtering=GCS_Filter()
nursechart=pd.read_csv("nurseCharting.csv",usecols=[1, 3, 4, 5, 7])
lab_in=pd.read_csv("lab.csv",usecols=[1, 2, 4, 5, 7, 8, 9])
respChart=pd.read_csv("respiratoryCharting.csv")
gcs_SOFA=gcs_filtering.extract_GCS_withSOFA(nursechart)
gcs_scores=gcs_filtering.extract_GCS(nursechart)
vent_details=gcs_filtering.extract_VENT(nursechart)
map_details=gcs_filtering.extract_MAP(nursechart)
lab_filtering=Lab_Filter()
lab_beforeSOFA=lab_filtering.extract_lab_format(lab_in, respChart, vent_details)

lab_withSOFA=lab_filtering.calc_lab_sofa(lab_beforeSOFA)

infdrug_filtering=Vasopressors()
infusionDrug=pd.read_csv("infusionDrug.csv")
patient_data=pd.read_csv("patient.csv", usecols=['patientunitstayid', 'admissionweight', 'dischargeweight', 'unitdischargeoffset'])
infusionfiltered=infdrug_filtering.extract_drugrates(infusionDrug)
normalized_infusion=infdrug_filtering.incorporate_weights(infusionfiltered, patient_data)
columnized_infusion=infdrug_filtering.add_separate_cols(normalized_infusion)
infusiondrug_withSOFA=infdrug_filtering.calc_SOFA(columnized_infusion,map_details)

sepsiscalc=tsepsis()
tsepsis_table=sepsiscalc.calc_tsepsis(lab_withSOFA, infusiondrug_withSOFA, gcs_SOFA, tsus_max_df)
table_merger=MergeTables()

vitals=pd.read_csv("vitalPeriodic.csv")
merged_table=table_merger.merge_final(gcs_scores, lab_beforeSOFA, infusiondrug_withSOFA, tsus_max_df, tsepsis_table, vitals)
merged_table.to_csv("merged_training_table.csv")

del merged_table
chunks=pd.read_csv("merged_training_table.csv", chunksize=1000000)
#Start sepsis predictions
sepsispred=SepsisPrediction()

#for loop to create all the necessary files, specify the time periods

time_prior = 2
time_duration = 6

num=1
for chunk in chunks:  
    sepsispred.process(chunk, num, time_prior, time_duration)
    num+=1

df = pd.read_csv('Sepsis'+str(time_prior)+'-'+str(time_duration)+str(1)+'.csv')
for i in range(num):
    df = pd.concat([df, pd.read_csv('Sepsis'+str(time_prior)+'-'+str(time_duration)+str(i+2)+'.csv')])    
df.reset_index(drop=True, inplace=True)

sepsis_df = sepsispred.case_preprocess(df)
labels = sepsis_df['label']
sepsis_X_train, sepsis_x_cv, sepsis_label, sepsis_y_cv = train_test_split(sepsis_df, labels, test_size=0.2, random_state=23)

controls_df = sepsispred.control_preprocess(df)
labels = controls_df['label']
X_train, x_cv, label, y_cv = train_test_split(controls_df, labels, test_size=0.2, random_state=23)

#adjust the parameters according to will
params = {'eta': 0.1, 'max_depth': 6, 'scale_pos_weight': 1, 'objective': 'reg:linear','subsample':0.25,'verbose': False}
xgb_model = None

sepsispred.run_xgboost(10, sepsis_X_train, sepsis_x_cv, sepsis_y_cv, X_train, x_cv, y_cv)