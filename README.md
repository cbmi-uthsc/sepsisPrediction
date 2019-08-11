<center><h2>Google Summer of Code - CBMI-UTHSC</h2></center>
<center><h3>Early Sepsis Prediction using Machine Learning</h3></center>
<h3>Table of Contents</h3>
<ol>
    <li>Introduction</li>
    <li>Modules</li>
    <li>Code Description</li>
    <li>GSoC Experience</li>
    <li>Conclusion</li>
</ol>

<h3>Introduction</h3>
<p>
This project aims to provide improved solution to the medical world, where millions of people die due to to Sepsis, a fatal disease where the patient has sustained and dyregulated response to infection. Since sepsis is time-sensitive, it quickly escalates to multiorgan failures, that greatly increases the risk of death. Here we try to accurately predict the occurence of sepsis hours before it actaully occurs. This will provide doctors to take contingency actions early, and will decrease mortality rates significantly.<br>

This project is based off the <b>eICU</b> database, managed by <b>physionet</b>. Critically ill patients are admitted to the ICU where they receive complex and time-sensitive care from a wide array of clinical staff. Electronic measuring devices are attached to them that produce data at regular intervals. This data, from multiple hospitals was assimilated into the eICU database.<br>
The vitals for the patients were measured every 5 minutes. Such a frequency is ideal because reduced frequency does not allow us to get a deep insght into the patient's condition, and consequentially, the models are not accurate enough.<br>

In this project, we apply multiple machine learning methods to generate descriptive features that are clinically meaningful and predict the onset of sepsis.
</p>

<i><b>NOTE: For the database features, please go through the documentation of the eICU database here: https://eicu-crd.mit.edu/about/eicu/</b></i>

<h3>Modules</h3>

<ol>
<li><h4>Extracting germane data</h4></li>
Since there are multiple tables to work with and the SOFA needs to be calculated from multiple sources, we converged all the relevant things to a single table. For reference, the following is the break-up (for debugging purposes):<br>

<ul>
    <li><b><i>lab.csv</i></b> was used to extract the lab values.</li>
    <li><b><i>nurseCharting.csv</i></b> was used to extract the GCS scores as well as the MAP and ventilator details.</li>
    <li><b><i>infusionDrug.csv</b></i> was used to extract all relevant vasopressors like Norepinephrine, Dopamine etc. </li>
    <li><b><i>vitalPeriodic.csv</i></b> was were all the vitals for the patients were recorded in a frequency of 5 minutes. </li>
    <li>The IV antibiotics data has been collected from the <b><i>medication.csv</i></b> table for each registered patient, while the fluid samples data was taken from the <b><i>microlab.csv</i></b></li>
    <li>Apart from the essential parameters needed for SOFA score calculation, we have also included a number of different variables to the final training data to check how they influence the model as will be shown in the feature importance curve. Some of them are:
    <ul>
    <li>calcium</li>
    <li>glucose</li>
    <li>lactate</li>
    <li>magnesium</li>
    <li>Phosphate</li>
    <li>potassium</li>
    </ul>
    </li>
</ul>
<br>
<li><h4>SOFA Calulations</h4></li>
For the SOFA calulation, we first merged all the aforementioned extracted tables. Then we followed the given rubrics to calculated the SOFA-3 scores.

![sofa calculation table](SOFA_CALC.png)

Here is a small code snippet of one of the parts of SOFA calculation:

<pre>
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
</pre>

<li><h4>Feature Extraction</h4></li>

<li><h4>Model Development (XGBoost and others)</h4></li>

</ol>
<h3>Code Description</h3>

<h3>GSoC Experience</h3>

<h3>Conclusion</h3>
