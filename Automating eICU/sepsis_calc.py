import pandas as pd
import numpy as np
import collections
import re
import time as time
import csv

class tsepsis:

    def calc_tsepsis(self, labs_withSOFA, df_cardiovascular, gcs_withSOFA, tsus_max):
        lab_sofa=pd.read_csv(labs_withSOFA)
        vent_sofa=pd.read_csv(df_cardiovascular,usecols=['patientunitstayid','offset','SOFA_cardio'])
        gcs_sofa=pd.read_csv(gcs_withSOFA,usecols=['patientunitstayid','offset','SOFA_GCS'])
        t_sus=pd.read_csv(tsus_max)

        patientIDs_IV = t_sus.patientunitstayid.unique().tolist()
        patientIDs_IV_sub = pd.DataFrame(patientIDs_IV)
        patientIDs_IV_sub.columns=['patientunitstayid']

        #Start merging

        labs_sus=pd.merge(lab_sofa,patientIDs_IV_sub,how="inner",on='patientunitstayid').drop_duplicates()
        vent_sus=pd.merge(vent_sofa,patientIDs_IV_sub,how="inner",on='patientunitstayid').drop_duplicates()
        gcs_sofa_sus=pd.merge(gcs_sofa,patientIDs_IV_sub,how="inner",on='patientunitstayid').drop_duplicates()
        labs_vent=pd.merge(labs_sus,vent_sus,on=['patientunitstayid','offset'],how="outer").drop_duplicates()
        final_sofa=pd.merge(labs_vent,gcs_sofa_sus,on=['patientunitstayid','offset'],how="outer").drop_duplicates()
        final_sofa=final_sofa.fillna(0)

        #Calculating the Total SOFA score, difference between scores and the cumulative time, from admission

        final_sofa=final_sofa.groupby(['patientunitstayid','offset'],as_index=False).max().drop_duplicates
        final_sofa=final_sofa.groupby(['patientunitstayid'],as_index=False).apply(pd.DataFrame.sort_values,'offset').reset_index()
        final_sofa=final_sofa.drop(columns=['level_0','level_1'])
        final_sofa['Total_SOFA']=final_sofa['SOFA_Coagulation']+final_sofa['SOFA_Liver']+final_sofa['SOFA_Respiration']+final_sofa['SOFA_Renal']+final_sofa['SOFA_cardio']+final_sofa['SOFA_GCS']
        del lab_sofa
        del vent_sofa
        del gcs_sofa

        #We need a way to check whether the time when there is a difference of 2 or more in SOFA score, is less than or equal to 24 hrs
        #Thus we either check (t_curr-t_min)<=24hrs or the better way, calculate the cumulative sum of diff of offsets
        final_sofa['diff_per_SOFA']=final_sofa.groupby(['patientunitstayid'])['Total_SOFA'].transform(lambda x: x.diff()).fillna(0)
        final_sofa['diff_per_offset']=final_sofa.groupby(['patientunitstayid'])['offset'].transform(lambda x:x.diff()).fillna(0)
        final_sofa['cumulative_time']=final_sofa.groupby(['patientunitstayid'])['diff_per_offset'].transform(lambda x:x.cumsum())

        #Filtering the SOFA table based on the (score diff >= 2 and cumulative time <= 24 hours)
        for_24_hr=final_sofa.loc[(final_sofa['diff_per_SOFA']>=2) & (final_sofa['cumulative_time']<=(24*60))]

        #Clubbing the t_suspicion table with filtered SOFA table
        t_sus=t_sus.rename(columns={'max':'tsus'})

        for_24_hr_tsofa=for_24_hr.groupby(['patientunitstayid']).agg({'offset':'min'}).reset_index()
        for_24_hr_tsofa=for_24_hr_tsofa.rename(columns={'offset':'tsofa'})
        for_24_hr_tsepsis=pd.merge(for_24_hr_tsofa,t_sus,on='patientunitstayid',how='inner').drop_duplicates()
        
        #flag==1 stands for cases, where as 0 for control.
        #Then we calculate the t_sepsis_onset time based on the required constraints

        for_24_hr_tsepsis['flag']=0
        for_24_hr_tsepsis.loc[(for_24_hr_tsepsis['tsofa']>=(for_24_hr_tsepsis['tsus']-(24*60))) & (for_24_hr_tsepsis['tsofa']<=(for_24_hr_tsepsis['tsus']+(12*60))),'flag']=1
        for_24_hr_tsepsis.to_csv("24_hour_sepsis.csv",index=False)
        
        #This following is to bring out the number of cases and the actual tsepsis, and not just flag them
        
        #for_24_hr_cases=for_24_hr_tsepsis[for_24_hr_tsepsis['flag']==1]
        #for_24_hr_cases['tsepsis']=for_24_hr_cases[['tsus','tsofa']].min(axis=1)

        return for_24_hr_tsepsis