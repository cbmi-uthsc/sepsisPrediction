import pandas as pd
import numpy as np
import collections
import re
import time as time
import csv

class Vasopressors:

    def extract_drugrates(self, infusionDrug):
        infusiondrug=infusionDrug
        infusiondrug['drugname']=infusiondrug['drugname'].astype('str')
        drug_names=infusiondrug['drugname'].unique()
        inf_drug=infusiondrug.loc[(infusiondrug['drugname'].str.contains("Dopamine")) | 
                              (infusiondrug['drugname'].str.contains("Norepinephrine")) | 
                              (infusiondrug['drugname'].str.contains("Epinephrine")) | 
                               (infusiondrug['drugname'].str.contains("norepinephrine")) | 
                               (infusiondrug['drugname'].str.contains("dopamine")) | 
                              (infusiondrug['drugname'].str.contains("Dobutamine"))]
        
        #Since the eICU database contains the drugnames as Name (Composition), we want to separate the composition and only keep the name
        inf_drug_new=inf_drug.loc[inf_drug['drugname'].str.contains("\(")]
        inf_drug_new['unit']=""
        
        #put the units into a separate columns
        inf_drug_new['unit']=inf_drug_new['drugname'].apply(lambda x: x.split('(')[1])

        inf_drug_filtered=inf_drug_new.loc[(inf_drug_new['unit'].str.contains("mcg/min")) | (inf_drug_new['unit'].str.contains("mcg/kg/min")) |
                (inf_drug_new['unit'].str.contains("mcg/hr")) | (inf_drug_new['unit'].str.contains("mg/hr")) |
                (inf_drug_new['unit'].str.contains("mcg/kg/hr")) | (inf_drug_new['unit'].str.contains("mg/min")) |
                (inf_drug_new['unit'].str.contains("mg/kg/min")) | (inf_drug_new['unit'].str.contains("nanograms/kg/min"))]

        
        #we remove all the garbage values (non-float values) from the drugrate to keep it purely numerical for further use
        inf_drug_filtered=inf_drug_filtered.loc[~(inf_drug_filtered['drugrate']=='OFF\\.br\\\\.br\\') & ~(inf_drug_filtered['drugrate']=='30\\.br\\') &
                     ~(inf_drug_filtered['drugrate']=='50 mcg/min') & ~(inf_drug_filtered['drugrate']=='50mcg/min\\.br\\') &
                     ~(inf_drug_filtered['drugrate']=='OFF') & ~(inf_drug_filtered['drugrate']=='Documentation undone')]
        
        inf_drug_filtered[['drugrate']]=inf_drug_filtered[['drugrate']].astype(str).astype(float)

        #Convert milli and nano to micro and hour to min to standardize units

        inf_drug_filtered.loc[inf_drug_filtered['unit'].str.contains("mg/hr"),'drugrate']=inf_drug_filtered['drugrate']*(100/6)
        inf_drug_filtered.loc[inf_drug_filtered['unit'].str.contains("mcg/hr"),'drugrate']=inf_drug_filtered['drugrate']*(1/60)
        inf_drug_filtered.loc[inf_drug_filtered['unit'].str.contains("mcg/kg/hr"),'drugrate']=inf_drug_filtered['drugrate']*(1/60)
        inf_drug_filtered.loc[inf_drug_filtered['unit'].str.contains("mg/min"),'drugrate']=inf_drug_filtered['drugrate']*(1000)
        inf_drug_filtered.loc[inf_drug_filtered['unit'].str.contains("mg/kg/min"),'drugrate']=inf_drug_filtered['drugrate']*(1000)
        inf_drug_filtered.loc[inf_drug_filtered['unit'].str.contains("nanograms/kg/min"),'drugrate']=inf_drug_filtered['drugrate']*(1/1000)

        inf_drug_filtered['unit']=inf_drug_filtered['unit'].replace(['mg/hr)','mcg/hr)','mg/min)','mcg/min)'],'mcg/min')
        inf_drug_filtered['unit']=inf_drug_filtered['unit'].replace(['mcg/kg/hr)','mg/kg/min)','nanograms/kg/min)','mcg/kg/min)'],'mcg/kg/min')

        return inf_drug_filtered

    def incorporate_weights(self, inf_drug_filtered, patient_data):

        #For patient_data, in eICU, the table is patient.csv, and the headers will be different for a different dataset
        patient=patient_data
        inf_drug_pre_patient=inf_drug_filtered
        
        inf_drug_pre_patient['admissionweight']=np.nan
        inf_drug_pre_patient['dischargeweight']=np.nan
        for pid in inf_drug_pre_patient['patientunitstayid'].unique():
            inf_drug_pre_patient.loc[inf_drug_pre_patient['patientunitstayid']==pid,'admissionweight']=patient.loc[patient['patientunitstayid']==pid,'admissionweight'].values[0]
            inf_drug_pre_patient.loc[inf_drug_pre_patient['patientunitstayid']==pid,'dischargeweight']=patient.loc[patient['patientunitstayid']==pid,'dischargeweight'].values[0]
        
        #save an additional copy before the normailzation stage
        inf_drug_pre_patient.to_csv("cardiovascular_params.csv", sep=',', index=False, encoding='utf-8')

        #As we can see, the records where admissionweight is 0, the discharge weight is also NaN. 
        #This means, we either omit these records, or fill the holes with a mean value. 
        #Since such records are very less as compared to entire dataset, we can fill them with their means. It won't skew the data much

        inf_drug_pre_patient['admissionweight']=inf_drug_pre_patient['admissionweight'].fillna(inf_drug_pre_patient['admissionweight'].mean())
        inf_drug_pre_patient['dischargeweight']=inf_drug_pre_patient['dischargeweight'].fillna(inf_drug_pre_patient['dischargeweight'].mean())

        #To avoid division by zero error, we will change the 0s for dischargeweight as 1
        inf_drug_pre_patient['dischargeweight']=inf_drug_pre_patient['dischargeweight'].replace([0,0.0],1)
        inf_drug_pre_patient.loc[((inf_drug_pre_patient['admissionweight']==np.nan) | (inf_drug_pre_patient['admissionweight']==0)) & 
                         (inf_drug_pre_patient['unit']=='mcg/min'),'drugrate']=inf_drug_pre_patient.loc[((inf_drug_pre_patient['admissionweight']==np.nan) | 
                          (inf_drug_pre_patient['admissionweight']==0)) & (inf_drug_pre_patient['unit']=='mcg/min')].apply(lambda x : (x['drugrate']/x['dischargeweight']),axis=1)
        inf_drug_pre_patient.loc[(inf_drug_pre_patient['admissionweight']>0) & 
                         (inf_drug_pre_patient['unit']=='mcg/min'),'drugrate']=inf_drug_pre_patient.loc[(inf_drug_pre_patient['admissionweight']>0) & 
                                                                                                        (inf_drug_pre_patient['unit']=='mcg/min')].apply(lambda x : (x['drugrate']/x['admissionweight']),axis=1)
        
        return inf_drug_pre_patient
    
    def add_separate_cols(self, inf_drug_weighted):
        inf_drug_pre_patient=inf_drug_weighted

        #renaming drugs to standardize changes made before in extract_drugrates() function
        inf_drug_pre_patient.loc[inf_drug_pre_patient['drugname'].str.contains('Norepi', case=True, regex=False),'drugname']='Norepinephrine'
        inf_drug_pre_patient.loc[inf_drug_pre_patient['drugname'].str.contains('Epi', case=True, regex=False),'drugname']='Epinephrine'
        inf_drug_pre_patient.loc[inf_drug_pre_patient['drugname'].str.contains('Dopa', case=False, regex=False),'drugname']='Dopamine'
        inf_drug_pre_patient.loc[inf_drug_pre_patient['drugname'].str.contains('Dobuta', case=True, regex=False),'drugname']='Dobutamine'

        #create columns
        inf_drug_pre_patient['Norepinephrine']=np.nan
        inf_drug_pre_patient['Epinephrine']=np.nan
        inf_drug_pre_patient['Dopamine']=np.nan
        inf_drug_pre_patient['Dobutamine']=np.nan

        #Feed the values from drugrate
        inf_drug_pre_patient['Norepinephrine']=inf_drug_pre_patient.loc[inf_drug_pre_patient['drugname']=='Norepinephrine','drugrate']
        inf_drug_pre_patient['Epinephrine']=inf_drug_pre_patient.loc[inf_drug_pre_patient['drugname']=='Epinephrine','drugrate']
        inf_drug_pre_patient['Dopamine']=inf_drug_pre_patient.loc[inf_drug_pre_patient['drugname']=='Dopamine','drugrate']
        inf_drug_pre_patient['Dobutamine']=inf_drug_pre_patient.loc[inf_drug_pre_patient['drugname']=='Dobutamine','drugrate']

        del inf_drug_pre_patient['infusiondrugid']
        del inf_drug_pre_patient['drugname']
        del inf_drug_pre_patient['infusionrate']
        del inf_drug_pre_patient['drugamount']
        
        #drugrate_norm is same as inf_drug_pre_patient
        inf_drug_pre_patient.to_csv("drugrate_norm.csv", sep=',', index=False, encoding='utf-8')

        return inf_drug_pre_patient

    def calc_SOFA(self, drugrate_norm, nursechartMAP):
        drugrate_updated=drugrate_norm
        columns=['patientunitstayid','infusionoffset','Norepinephrine','Epinephrine','Dopamine','Dobutamine']
        drugrate_updated=drugrate_updated[columns]

        #from earlier gcs_extract file
        nursemap=nursechartMAP
        nursemap=nursemap[['patientunitstayid','nursingchartentryoffset','nursingchartvalue']]
        #since we need both the cardio params as well as the MAP values individually, we will use outer join
        df_cardiovascular=pd.merge(drugrate_updated,nursemap,left_on=['patientunitstayid'],right_on=['patientunitstayid'],how='outer').drop_duplicates()
        
        df_cardiovascular.loc[((df_cardiovascular['Dopamine'] >15)) | ((df_cardiovascular['Epinephrine'] >0.1)) | ((df_cardiovascular['Norepinephrine'] >0.1)), 'SOFA_cardio'] = 4
        df_cardiovascular.loc[((df_cardiovascular['Dopamine'] <15)) | ((df_cardiovascular['Epinephrine'] <=0.1)) | ((df_cardiovascular['Norepinephrine'] <=0.1)), 'SOFA_cardio'] = 3
        df_cardiovascular.loc[((df_cardiovascular['Dopamine'] <5)) | ((df_cardiovascular['Dobutamine'] >0)), 'SOFA_cardio'] = 2
        df_cardiovascular.loc[df_cardiovascular['nursingchartvalue']<70,'SOFA_cardio']=1
        df_cardiovascular.loc[df_cardiovascular['nursingchartvalue']>71,'SOFA_cardio']=0

        #Now we need to combine these 2 timeoffsets into a single offset
        df_cardiovascular['offset'] = np.where(df_cardiovascular['infusionoffset']>0, df_cardiovascular['infusionoffset'], df_cardiovascular['nursingchartentryoffset'])
        df_cardiovascular = df_cardiovascular.dropna(subset = ['offset'])

        del df_cardiovascular['infusionoffset']
        del df_cardiovascular['nursingchartentryoffset']
        df_cardiovascular['offset']=df_cardiovascular['offset'].astype('int32')

        df_cardiovascular.to_csv("drugate_norm_updated.csv",index=False)

        return df_cardiovascular



