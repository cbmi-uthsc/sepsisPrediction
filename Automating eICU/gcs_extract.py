import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time as time
import csv

class GCS_Filter:
    # functions for extracting gcs scores, map values, and ventilator details
    # input for all functions is the "nurseCharting" table in the eICU database. For custom input, first convert your input into the suitable format
    
    def extract_GCS_withscores(self, nurseCharting):
        l_nursechart=[]
        for chunk in pd.read_csv(nurseCharting, chunksize=10000, usecols=[1, 3, 4, 5, 7]):
            df=chunk.loc[(chunk['nursingchartcelltypecat']=='Scores') & (chunk['nursingchartcelltypevallabel']=='Glasgow coma score')]
            l_nursechart.append(df)
        del chunk
        del df
        df_nursechart=pd.concat(l_nursechart, sort=False)
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
        l_nursechart=[]
        for chunk in pd.read_csv(nurseCharting, chunksize=10000, usecols=[1, 3, 4, 5, 7]):
            df=chunk.loc[(chunk['nursingchartcelltypecat']=='Scores') & (chunk['nursingchartcelltypevallabel']=='Glasgow coma score')]
            l_nursechart.append(df)
        del chunk
        del df
        df_nursechart=pd.concat(l_nursechart, sort=False)
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
        l_map=[]
        for chunk in pd.read_csv(nurseCharting, chunksize=10000, usecols=[1, 3, 4, 5, 7]):
            df=chunk.loc[(chunk['nursingchartcelltypevallabel']=='MAP (mmHg)')]
            l_map.append(df)
        del chunk
        del df
        df_map=pd.concat(l_map,sort=False)
        df_map.to_csv("nursechartMAP.csv",index=False)
        
        return df_map

        
    def extract_VENT(self, nurseCharting):
        l_vent=[]
        for chunk in pd.read_csv(nurseCharting, chunksize=10000, usecols=[1, 3, 4, 5, 7]):
            df=chunk.loc[(chunk['nursingchartcelltypevallabel']=='O2 Admin Device')]
            l_vent.append(df)
        del chunk
        del df
        df_vent=pd.concat(l_vent,sort=False)
        df_vent.to_csv("nursechartVent.csv",index=False)
        
        return df_vent