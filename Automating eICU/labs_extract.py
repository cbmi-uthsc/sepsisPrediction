import pandas as pd
import numpy as np
import collections
import re
import time as time
import csv

class Lab_filter:
    
    def extract_lab_format(self, lab, respChart, nursechartVent):
        l_labs=[]
        for chunk in pd.read_csv(lab, chunksize=10000, usecols=[1, 2, 4, 5, 7, 8, 9]):
            l_labs.append(chunk)
        del chunk

        df_labs=pd.concat(l_labs, sort=False)
        df_labs.to_csv("labs_before_FiO2.csv", sep=',', index=False, encoding='utf-8')

        respiratory=pd.read_csv(respChart, chunksize=10000)
        l_resp=[]
        for chunk in respiratory:
            l_resp.append(chunk)  
        
        df_resp=pd.concat(l_resp, sort=False)
        del chunk
        del l_resp
        
        respFiO2=df_resp.loc[df_resp['respchartvaluelabel']=='FiO2']
        respFiO2.loc[respFiO2['respchartvalue']==0, 'respchartvalue']=1
        respFiO2['respchartvalue']=respFiO2['respchartvalue'].str.replace('%','')
        respFiO2=respFiO2[['patientunitstayid','respchartentryoffset', 'respchartvaluelabel', 'respchartvalue']]
        respFiO2[['patientunitstayid','respchartentryoffset']]=respFiO2[['patientunitstayid','respchartentryoffset']].astype('int32')
        respFiO2[['respchartvalue']]=respFiO2[['respchartvalue']].astype('str').astype('float32')

        df_labs['paO2_FiO2']=np.nan
        df_labs=df_labs[['patientunitstayid', 'labresultoffset', 'labname', 'labresult', 'paO2_FiO2' ]]
        df_labs[['patientunitstayid','labresultoffset']]=df_labs[['patientunitstayid','labresultoffset']].astype('int32')
        df_labs[['labresult','paO2_FiO2']]=df_labs[['labresult','paO2_FiO2']].astype('float32')

    
        for row in df_labs.itertuples():
            if (row.labname=='paO2'):
                pid=row.patientunitstayid
                toffset=row.labresultoffset
                index=row.Index
                if(any(respFiO2.loc[(respFiO2['patientunitstayid']==pid) & ((respFiO2['respchartentryoffset']<=toffset+200) & (respFiO2['respchartentryoffset']>=toffset-200)),'respchartvalue'].values)):
                    result = row.labresult/(respFiO2.loc[(respFiO2['patientunitstayid']==pid) & ((respFiO2['respchartentryoffset']<=toffset+200) & (respFiO2['respchartentryoffset']>=toffset-200)),'respchartvalue'].iloc[0]/100)
                    df_labs['paO2_FiO2'][index]=result      
                else:
                    df_labs['paO2_FiO2'][index]=df_labs['paO2_FiO2'].mean()
        
        df_labs.to_csv("labs_after_FiO2.csv", sep=',', index=False, encoding='utf-8')

        del df_labs

        pre_final_lab=pd.read_csv("labs_after_FiO2.csv")
        pre_final_lab['platelets_x_1000']=np.nan
        pre_final_lab['total_bilirubin']=np.nan
        pre_final_lab['urinary_creatinine']=np.nan
        pre_final_lab['creatinine']=np.nan


        pre_final_lab['platelets_x_1000']=pre_final_lab.loc[pre_final_lab['labname']=='platelets x 1000','labresult']
        pre_final_lab['total_bilirubin']=pre_final_lab.loc[pre_final_lab['labname']=='total bilirubin','labresult']
        pre_final_lab['urinary_creatinine']=pre_final_lab.loc[pre_final_lab['labname']=='urinary creatinine','labresult']
        pre_final_lab['creatinine']=pre_final_lab.loc[pre_final_lab['labname']=='creatinine','labresult']

        pre_final_lab.loc[pre_final_lab['labname'].str.contains('HCO3'),'HCO3']=pre_final_lab['labresult']
        pre_final_lab.loc[pre_final_lab['labname'].str.contains('pH'),'pH']=pre_final_lab['labresult']
        pre_final_lab.loc[pre_final_lab['labname'].str.contains('paCO2',case=False),'paCO2']=pre_final_lab['labresult']
        pre_final_lab.loc[pre_final_lab['labname'].str.contains('direct',case=False),'direct_bilirubin']=pre_final_lab['labresult']
        pre_final_lab.loc[pre_final_lab['labname'].str.contains('excess',case=False),'excess']=pre_final_lab['labresult']
        pre_final_lab.loc[pre_final_lab['labname'].str.contains('ast',case=False),'ast']=pre_final_lab['labresult']
        pre_final_lab.loc[pre_final_lab['labname'].str.contains('bun',case=False),'bun']=pre_final_lab['labresult']
        pre_final_lab.loc[pre_final_lab['labname'].str.contains('Calcium',case=False),'calcium']=pre_final_lab['labresult']
        pre_final_lab.loc[pre_final_lab['labname'].str.contains('glucose',case=False),'glucose']=pre_final_lab['labresult']
        pre_final_lab.loc[pre_final_lab['labname'].str.contains('lactate',case=False),'lactate']=pre_final_lab['labresult']
        pre_final_lab.loc[pre_final_lab['labname'].str.contains('magnesium',case=False),'magnesium']=pre_final_lab['labresult']
        pre_final_lab.loc[pre_final_lab['labname'].str.contains('phosphate',case=False),'phosphate']=pre_final_lab['labresult']
        pre_final_lab.loc[pre_final_lab['labname'].str.contains('potassium',case=False),'potassium']=pre_final_lab['labresult']
        pre_final_lab.loc[pre_final_lab['labname'].str.contains('hct',case=False),'hct']=pre_final_lab['labresult']
        pre_final_lab.loc[pre_final_lab['labname'].str.contains('hgb',case=False),'hgb']=pre_final_lab['labresult']
        pre_final_lab.loc[pre_final_lab['labname'].str.contains('ptt',case=False),'ptt']=pre_final_lab['labresult']
        pre_final_lab.loc[pre_final_lab['labname'].str.contains('WBC x 1000'),'wbc']=pre_final_lab['labresult']
        pre_final_lab.loc[pre_final_lab['labname'].str.contains('fibrinogen',case=False),'fibrinogen']=pre_final_lab['labresult']
        pre_final_lab.loc[pre_final_lab['labname'].str.contains('troponin',case=False),'troponin']=pre_final_lab['labresult']
        


        pre_final_lab['paO2_FiO2']=pre_final_lab['paO2_FiO2'].astype('float32')
        pre_final_lab['platelets_x_1000']=pre_final_lab['platelets_x_1000'].astype('float32')
        pre_final_lab['total_bilirubin']=pre_final_lab['total_bilirubin'].astype('float32')
        pre_final_lab['urinary_creatinine']=pre_final_lab['urinary_creatinine'].astype('float32')

        pre_final_lab=pre_final_lab.replace([np.inf, -np.inf], np.nan)

        del pre_final_lab['labresult']
        del pre_final_lab['labname']


        #Merge the ventilator details with the lab data for SOFA calculations
        nursevent=pd.read_csv(nursechartVent)
        nursevent=nursevent[['patientunitstayid','nursingchartentryoffset','nursingchartvalue']]
        labs_withO2=pd.merge(pre_final_lab,nursevent,left_on=['patientunitstayid'],right_on=['patientunitstayid'],how='left').drop_duplicates()
        
        return labs_withO2
        
        
    def calc_lab_sofa(self, labs_withO2):

        labs_withO2=pd.read_csv(labs_withO2)

        labs_withO2.loc[(labs_withO2['platelets_x_1000'] >=150), 'SOFA_Coagulation'] = 0
        labs_withO2.loc[(labs_withO2['platelets_x_1000'] <150), 'SOFA_Coagulation'] = 1
        labs_withO2.loc[(labs_withO2['platelets_x_1000'] <100) , 'SOFA_Coagulation'] = 2
        labs_withO2.loc[(labs_withO2['platelets_x_1000'] <50), 'SOFA_Coagulation'] = 3
        labs_withO2.loc[(labs_withO2['platelets_x_1000'] <20), 'SOFA_Coagulation'] = 4

        labs_withO2.loc[(labs_withO2['total_bilirubin'] <1.2), 'SOFA_Liver'] = 0
        labs_withO2.loc[(labs_withO2['total_bilirubin'] >=1.2) & (labs_withO2['total_bilirubin'] <=1.9), 'SOFA_Liver'] = 1
        labs_withO2.loc[(labs_withO2['total_bilirubin'] >=2) & (labs_withO2['total_bilirubin'] <=5.9), 'SOFA_Liver'] = 2
        labs_withO2.loc[(labs_withO2['total_bilirubin'] >=6) & (labs_withO2['total_bilirubin'] <=11.9), 'SOFA_Liver'] = 3
        labs_withO2.loc[(labs_withO2['total_bilirubin'] >12), 'SOFA_Liver'] = 4

        labs_withO2.loc[(labs_withO2['paO2_FiO2'] >=400), 'SOFA_Respiration'] = 0
        labs_withO2.loc[(labs_withO2['paO2_FiO2'] <400), 'SOFA_Respiration'] = 1
        labs_withO2.loc[(labs_withO2['paO2_FiO2'] <300), 'SOFA_Respiration'] = 2
        labs_withO2.loc[((labs_withO2['paO2_FiO2'] <200) & (labs_withO2['nursingchartvalue'] =='ventilator')), 'SOFA_Respiration'] = 3
        labs_withO2.loc[((labs_withO2['paO2_FiO2'] <100) & (labs_withO2['nursingchartvalue'] =='ventilator')), 'SOFA_Respiration'] = 4

        labs_withO2.loc[((labs_withO2['creatinine'] >=0) & (labs_withO2['creatinine'] <=1.1)), 'SOFA_Renal'] = 0
        labs_withO2.loc[((labs_withO2['creatinine'] >=1.2) & (labs_withO2['creatinine'] <=1.9)), 'SOFA_Renal'] = 1
        labs_withO2.loc[((labs_withO2['creatinine'] >=2) & (labs_withO2['creatinine'] <=3.4)), 'SOFA_Renal'] = 2
        labs_withO2.loc[((labs_withO2['creatinine'] >=3.5) & (labs_withO2['creatinine'] <=4.9)) | (labs_withO2['urinary_creatinine'] <200), 'SOFA_Renal'] = 3
        labs_withO2.loc[(labs_withO2['creatinine'] >5) | (labs_withO2['urinary_creatinine'] <200), 'SOFA_Renal'] = 4

        labs_withO2['offset'] = np.where(labs_withO2['labresultoffset']>0, labs_withO2['labresultoffset'], labs_withO2['nursingchartentryoffset'])
        labs_withO2[['SOFA_Coagulation','SOFA_Liver','SOFA_Respiration','SOFA_Renal']]=labs_withO2[['SOFA_Coagulation','SOFA_Liver','SOFA_Respiration','SOFA_Renal']].fillna(0)
        labs_SOFA=labs_withO2[['patientunitstayid','offset','SOFA_Coagulation','SOFA_Liver','SOFA_Respiration','SOFA_Renal']].drop_duplicates()
        labs_SOFA.to_csv('Labs_withSOFA.csv',index=False)

        return labs_SOFA

                    

            
    
    
    
    

