import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time as time
import csv


class GCS_Filter:
    # functions for extracting gcs scores, map values, and ventilator details
    # input for all functions is the "nurseCharting" table in the eICU database. For custom input, first convert your input into the suitable format
    
    def extract_GCS_withscores(self, nurseCharting):
        
        df_nursechart=nurseCharting.loc[(nurseCharting['nursingchartcelltypecat']=='Scores') & (nurseCharting['nursingchartcelltypevallabel']=='Glasgow coma score')]
        df_nursechart.to_csv("gcs_scores.csv",sep=',',index=False,encoding='utf-8')
        df_gcs=df_nursechart
        del df_nursechart
        df_gcs['GCS_Score']=np.nan
        df_gcs=df_gcs.loc[~df_gcs['nursingchartvalue'].str.contains("unable",case=False,na=False)]
        df_gcs[['GCS_Score']]=df_gcs[['nursingchartvalue']]
        del df_gcs['nursingchartvalue']
        df_gcs.to_csv("gcs_scores_updated.csv",sep=',',index=False,encoding='utf-8')
        
        d_gcs=pd.read_csv('gcs_scores_updated.csv')
        columns=['patientunitstayid','nursingchartentryoffset','GCS_Score']
        d_gcs=d_gcs[columns]
        d_gcs=d_gcs.dropna()

        d_gcs.loc[d_gcs['GCS_Score']>=15,'SOFA_GCS']=0
        d_gcs.loc[d_gcs['GCS_Score']<15,'SOFA_GCS']=1
        d_gcs.loc[d_gcs['GCS_Score']<13,'SOFA_GCS']=2
        d_gcs.loc[d_gcs['GCS_Score']<10,'SOFA_GCS']=3
        d_gcs.loc[d_gcs['GCS_Score']<6,'SOFA_GCS']=4

        d_gcs=d_gcs.rename(columns={'nursingchartentryoffset':'offset'})
        d_gcs.to_csv("gcs_withSOFA.csv",index=False)
        
        return d_gcs

    def extract_GCS(self, nurseCharting):
        df_nursechart=nurseCharting.loc[(nurseCharting['nursingchartcelltypecat']=='Scores') & (nurseCharting['nursingchartcelltypevallabel']=='Glasgow coma score')]
        df_nursechart.to_csv("gcs_scores.csv",sep=',',index=False,encoding='utf-8')
        df_gcs=df_nursechart
        del df_nursechart
        df_gcs['GCS_Score']=np.nan
        df_gcs=df_gcs.loc[~df_gcs['nursingchartvalue'].str.contains("unable",case=False,na=False)]
        df_gcs[['GCS_Score']]=df_gcs[['nursingchartvalue']]
        del df_gcs['nursingchartvalue']
        df_gcs.to_csv("gcs_scores_updated.csv",sep=',',index=False,encoding='utf-8')

        return df_gcs

        
    
    def extract_MAP(self, nurseCharting):
        df_map=nurseCharting.loc[(nurseCharting['nursingchartcelltypevallabel']=='MAP (mmHg)')]
        df_map.to_csv("nursechartMAP.csv",index=False)
        
        return df_map

        
    def extract_VENT(self, nurseCharting):
        df_vent=nurseCharting.loc[(nurseCharting['nursingchartcelltypevallabel']=='O2 Admin Device')]
        df_vent.to_csv("nursechartVent.csv",index=False)
        
        return df_vent