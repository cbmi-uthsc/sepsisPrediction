import pandas as pd
import numpy as np
import collections
import re


class tsuspicion:
    #Input needs to be the table containing drug data, treatment data and microlab data
    
    def get_antibiotics(self, medication_table_in, treatment_table_in, microlab_input):
        medication=medication_table_in
        # subselect the variable columns
        medication_columns = ['drugorderoffset', 'patientunitstayid','drugstartoffset','drugname', 'dosage', 'routeadmin', 'drugstopoffset']
        medication_subset = medication[medication_columns]
        #medication_subset = medication_subset[medication_subset['routeadmin']== 'IV']
        del medication
        
        antibiotics_list=['adoxa','ala-tet','alodox','amikacin','amikin','amoxicillin','clavulanate','ampicillin',
                         'augmentin','avelox','avidoxy','azactam','azithromycin','aztreonam','axetil','bactocill',
                         'bactrim','bethkis','biaxin','bicillin l-a','cayston','cefazolin','cedax','cefoxitin',
                         'ceftazidime','cefaclor','cefadroxil','cefdinir','cefditoren','cefepime','cefotetan','cefotaxime',
                         'cefpodoxime','cefprozil','ceftibuten','ceftin','cefuroxime','cefuroxime','cephalexin','chloramphenicol',
                         'cipro','ciprofloxacin','claforan','clarithromycin','cleocin','clindamycin','cubicin','dicloxacillin',
                         'doryx','doxycycline','duricef','dynacin','ery-tab','eryped','eryc','erythrocin','erythromycin','factive',
                         'flagyl','fortaz','furadantin','garamycin','gentamicin','kanamycin','keflex','ketek','levaquin',
                         'levofloxacin','lincocin','macrobid','macrodantin','maxipime','mefoxin','metronidazole','minocin',
                         'minocycline','monodox','monurol','morgidox','moxatag','moxifloxacin','myrac','nafcillin sodium',
                         'nicazel doxy 30','nitrofurantoin','noroxin','ocudox','ofloxacin','omnicef','oracea','oraxyl','oxacillin',
                         'pc pen vk','pce dispertab','panixine','pediazole','penicillin','periostat','pfizerpen','piperacillin',
                         'tazobactam','primsol','proquin','raniclor','rifadin','rifampin','rocephin','smz-tmp','septra','septra ds',
                         'septra','solodyn','spectracef','streptomycin sulfate','sulfadiazine','sulfamethoxazole','trimethoprim',
                         'sulfatrim','sulfisoxazole','suprax','synercid','tazicef','tetracycline','timentin','tobi','tobramycin',
                         'trimethoprim','unasyn','vancocin','vancomycin','vantin','vibativ','vibra-tabs','vibramycin','zinacef',
                         'zithromax','zmax','zosyn','zyvox']

        medication_subset_filt = medication_subset[medication_subset.drugname.str.contains('|'.join(antibiotics_list), flags=re.IGNORECASE, na=False)]
        del medication_subset
        
        treatment=treatment_table_in
        treatment_columns = ['treatmentoffset', 'patientunitstayid','treatmentstring']
        treatment_subset = treatment[treatment_columns]
        treatment_subset = treatment_subset[treatment_subset['treatmentstring'].str.contains("cardiovascular")]
        treatment_columns = ['treatmentoffset', 'patientunitstayid']
        treatment_subset = treatment_subset[treatment_columns]
        treatment_subset_new = treatment_subset.rename(columns={'treatmentoffset': 'culturetakenoffset'})

        microlab=microlab_input
        microlab_columns = ['culturetakenoffset', 'patientunitstayid']
        microlab_subset = microlab[microlab_columns]
        treatment_microlab = pd.concat([treatment_subset_new,microlab_subset])

        ABX_BC_pid_offset = pd.merge(medication_subset_filt[['patientunitstayid', 'drugorderoffset']], treatment_microlab[['patientunitstayid', 'culturetakenoffset']], on='patientunitstayid', how='inner').drop_duplicates()
        # For each PID, the difference of each blood culture and each antibiotics time is calculated 
        ABX_BC_pid_offset.loc[(ABX_BC_pid_offset['drugorderoffset'] < ABX_BC_pid_offset['culturetakenoffset']) & (ABX_BC_pid_offset['culturetakenoffset'] - ABX_BC_pid_offset['drugorderoffset']  <=24*3600), 'tSuspicion'] = pd.DataFrame([ABX_BC_pid_offset['culturetakenoffset'], ABX_BC_pid_offset['drugorderoffset']]).min()
        ABX_BC_pid_offset.loc[(ABX_BC_pid_offset['culturetakenoffset'] < ABX_BC_pid_offset['drugorderoffset']) & (ABX_BC_pid_offset['drugorderoffset'] - ABX_BC_pid_offset['culturetakenoffset']  <=72*3600), 'tSuspicion'] = pd.DataFrame([ABX_BC_pid_offset['drugorderoffset'], ABX_BC_pid_offset['culturetakenoffset']]).min()

        ABX_BC_pid_offset.drop_duplicates(inplace=True)
        ABX_BC_pid_offset_Clean = ABX_BC_pid_offset[np.isfinite(ABX_BC_pid_offset['tSuspicion'])]
        ABX_BC_pid_offset_Clean.to_csv("ABX_BC_pid_offset_Clean_Tsuspicion.csv", sep=',', index=False)

        suspicion = pd.read_csv("ABX_BC_pid_offset_Clean_Tsuspicion.csv")
        suspicionUnique = suspicion[['patientunitstayid', 'tSuspicion']].drop_duplicates()
        suspicionUnique.to_csv("ABX_BC_pid_offset_Clean_TsuspicionUnique.csv", sep=',', index=False)
        suspicionMax = suspicion.groupby(['patientunitstayid']).agg({'tSuspicion': ['max']})
        suspicionMax.columns = suspicionMax.columns.droplevel(0)
        suspicionMax.to_csv("ABX_BC_pid_offset_Clean_TsuspicionMax.csv", sep=',')

        patientIDs_IV = suspicionUnique.patientunitstayid.unique().tolist()
        patientIDs_IV_sub = pd.DataFrame(patientIDs_IV)
        patientIDs_IV_sub.columns=['patientunitstayid']
        del medication_subset_filt
        del antibiotics_list
        
        return suspicionMax