import pandas as pd
import numpy as np
from antibiotics import tsuspicion
from gcs_extract import GCS_Filter
from labs_extract import Lab_Filter
from merge_final_table import MergeTables
from sepsis_calc import tsepsis
from vasopressor_extract import Vasopressors
from sepsisprediction import SepsisPrediction
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
gcs_scores=gcs_filtering.extract_GCS_withscores(nursechart)
vent_details=gcs_filtering.extract_VENT(nursechart)
map_details=gcs_filtering.extract_MAP(nursechart)
lab_filtering=Lab_Filter()
#lab_beforeSOFA=lab_filtering.extract_lab_format(lab_in, respChart, vent_details)

#lab_withSOFA=lab_filtering.calc_lab_sofa(lab_beforeSOFA)

infdrug_filtering=Vasopressors()
infusionDrug=pd.read_csv("infusionDrug.csv")
patient_data=pd.read_csv("patient.csv", usecols=['patientunitstayid', 'admissionweight', 'dischargeweight', 'unitdischargeoffset'])
infusionfiltered=infdrug_filtering.extract_drugrates(infusionDrug)
normalized_infusion=infdrug_filtering.incorporate_weights(infusionfiltered, patient_data)
columnized_infusion=infdrug_filtering.add_separate_cols(normalized_infusion)
infusiondrug_withSOFA=infdrug_filtering.calc_SOFA(columnized_infusion,map_details)
