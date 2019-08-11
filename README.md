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
<li>Merging Tables</li>
Since there are multiple tables to work with and the SOFA needs to be calculated from multiple sources, we converged all the relevant things to a single table. For reference, the following is the break-up (for debugging purposes):<br>
<pre>
<ul>
    <li><b><i>lab.csv</i></b> was used to extract the lab values.</li>
    <li><b><i>nurseCharting.csv</i></b> was used to extract the GCS scores as well as the MAP and ventilator details.</li>
    <li><b><i>infusionDrug.csv</b></i> was used to extract all relevant vasopressors like Norepinephrine, Dopamine etc. </li>
    <li></li>
</ul>
</pre>

<li>SOFA Calulations</li>
<li>Feature Extraction</li>
<li>Model Development (XGBoost and others)</li>
</ol>
<h3>Code Description</h3>

<h3>GSoC Experience</h3>

<h3>Conclusion</h3>
