import pandas as pd
import numpy as np
import collections
import re
import time as time
import csv
import os
import glob

class MergeTables:

    def merge_final(self, gcs_scores_updated, labs_morevars, drugrate_norm_updated, tsus_max, tsepsis_table):

        d_gcs=gcs_scores_updated
        d_gcs=d_gcs.drop(columns=['nursingchartcelltypecat','nursingchartcelltypevallabel'])
        d_gcs=d_gcs.drop_duplicates()
        d_gcs=d_gcs.rename(columns={'nursingchartentryoffset':'offset'})

        labs=labs_morevars
        labs=labs.drop_duplicates()
        labs=labs.rename(columns={'labresultoffset':'offset'})

        cardiovas=drugrate_norm_updated
        cardiovas.drop_duplicates()
        cardiovas=cardiovas.drop(columns=['nursingchartvalue','SOFA_cardio'])

        tsus=tsus_max
        tsus=tsus.rename(columns={'max':'offset'})

        sepsis_labels=tsepsis_table
        sepsis_labels['tsepsis']=sepsis_labels[['tsus','tsofa']].min(axis=1)

        #First merge
        labs_cardio=pd.merge(labs,cardiovas,how="outer",on=['patientunitstayid','offset']).drop_duplicates()
        #labs_cardio.to_csv("labs_cardio_interim.csv",index=False)

        #del labs
        #del cardiovas

        training_build=pd.merge(labs_cardio,d_gcs,how="outer",on=['patientunitstayid','offset']).drop_duplicates()
        #del d_gcs

        #To correct the replication of offsets for same patients
        training_build=training_build.groupby(['patientunitstayid','offset'],as_index=False).max().drop_duplicates()

        #To get offsets from the suspicion table
        training_build=pd.merge(training_build,tsus,on=['patientunitstayid','offset'],how='outer')

        training_build=training_build.groupby(['patientunitstayid'],as_index=False).apply(pd.DataFrame.sort_values,'offset').reset_index()
        training_build=training_build.drop(columns=['level_0','level_1'])
        training_build=training_build.drop(columns=['Norepinephrine','Epinephrine','Dopamine','Dobutamine'])
        #training_build_filtered=training_build.dropna(subset=['paO2_FiO2','platelets_x_1000','total_bilirubin','urinary_creatinine','creatinine','HCO3','pH','paCO2','direct_bilirubin','excess','ast','bun','calcium','GCS_Score'],how='all')
        training_build['label']=np.nan

        final_build=pd.merge(training_build,sepsis_labels,how='outer',left_on=['patientunitstayid','offset'],right_on=['patientunitstayid','tsepsis']).drop_duplicates()
        final_build.label=final_build.flag
        final_build=final_build.drop(columns=['tsofa','tsus','tsepsis','flag'])
        #After the initial sepsis=1 flag, all the labels for that patient is given label=1, all before that is 0
        final_build['label']=final_build.groupby(['patientunitstayid'])['label'].ffill()
        final_build['label']=final_build['label'].fillna(0)
        final_build=final_build.drop_duplicates()

        del training_build
        del d_gcs
        del labs
        del labs_cardio

        #we need to add vitals as well for these patients
        pids=final_build['patientunitstayid'].unique()
        vitals=pd.read_csv("vitalPeriodic.csv",usecols=['patientunitstayid','observationoffset','heartrate','respiration'])
        tomerge=vitals.loc[vitals['patientunitstayid'].isin(pids)]
        del vitals
        tomerge=tomerge.rename(columns={'observationoffset':'offset'})
        train_withvitals=pd.merge(final_build,tomerge,on=['patientunitstayid','offset'],how='outer')
        del tomerge
        train_withvitals=train_withvitals.replace([np.inf, -np.inf], np.nan)

        train_withvitals=train_withvitals.groupby(['patientunitstayid','offset'],as_index=False).max().drop_duplicates()
        train_withvitals=train_withvitals.groupby(['patientunitstayid'],as_index=False).apply(pd.DataFrame.sort_values,'offset').reset_index()
        train_withvitals=train_withvitals.drop(columns=['level_0','level_1'])
        train_withvitals=train_withvitals.dropna(subset=['offset','patientunitstayid'])

        columns=list(train_withvitals.columns)
        columns.remove('patientunitstayid')
        columns.remove('offset')
        adjusted_cols=columns

        #one forward pass to fill NAs inside each patientid group
        train_withvitals=train_withvitals.groupby(['patientunitstayid']).ffill()

        #backfill all the labels to avoid confusion
        train_withvitals['label']=train_withvitals.groupby(['patientunitstayid'])['label'].bfill()

        #fill remaining NA values with the median values of respective columns
        train_withvitals=train_withvitals.fillna(train_withvitals.median())

        train_withvitals=train_withvitals.dropna(subset=adjusted_cols,how='all')

        train_withvitals.to_csv("final_training_table.csv",index=False)

        return train_withvitals