# ICU Mortality Risk Prediction Model

This repository contains a minimal inference pipeline 
for mortality risk prediction on Covid and non-Covid patients in the intensive care unit (ICU) using an XGBoost model.
Data collection and model development was a collaboration between Unetiq GmbH and Universitätsklinikum Regensburg (UKR) 
as part of the NUM CODEX+ project.

## Background

Mortality prediction is a crucial part of the risk assessment of ICU patients to enable rapid decision-making.
Conventional risk scores such as SOFA or APACHE II often have limited predictive power, 
especially on Covid patients that increasingly needed ICU treatment during the Covid pandemic.
For this model, top predictive ICU measurement features have been identified through feature selection and importance analysis
on both Covid and non-Covid patient cohorts. 
The model significantly outperforms APACHE II (AUC 0.64) and SOFA (0.6) scores on Covid patients.


## Requirements


| Requirement  | Version | Note                                                                  |
|--------------|---------|-----------------------------------------------------------------------|
| Java         | 11.0.7  | OpenJDK recommended                                                   |
| Apache Maven | 3.6.0   | Dependency management                                                 |
| xgboost4j    | 1.6.2   | Automatic install during Maven compilation |

To validate the requirements, check if the output of the following commands are similar to this:   

```bash
unetiq@codex:~/mortality-ukr$ java --version
openjdk 11.0.17 2022-10-18
OpenJDK Runtime Environment (build 11.0.17+8-post-Ubuntu-1ubuntu218.04)
OpenJDK 64-Bit Server VM (build 11.0.17+8-post-Ubuntu-1ubuntu218.04, mixed mode, sharing)
```

```bash
unetiq@codex:~/mortality-ukr$ mvn --version
Apache Maven 3.6.0
Maven home: /usr/share/maven
Java version: 11.0.17, vendor: Ubuntu, runtime: /usr/lib/jvm/java-11-openjdk-amd64
Default locale: en, platform encoding: UTF-8
OS name: "linux", version: "5.4.0-1094-azure", arch: "amd64", family: "unix"
```

The code was developed and tested on a Linux (Ubuntu 18.04) system.

## Installation

Assuming the requirements are installed, from the root directory run

```bash
sh build.sh
```

This will resolve dependencies, compile and package the project, resulting in a ```.jar``` file in ```target/mortality-ukr-1.0-SNAPSHOT.jar```. The jar file is subsequently executed to read the example ICU data and save the model prediction outcomes. Note that example result files are already provided and will be regenerated when running the project.

After initial installation, you can run the compiled code without re-installing it via

```bash
java -cp target/mortality-ukr-1.0-SNAPSHOT.jar de.unetiq.RiskScoreICU
```


## Usage

### Data

Exemplary ICU measurements for 10 Covid and 10 non-Covid patients aggregated over the first 24 hours upon their addmission to the ICU are provided in JSON format in ```src/main/resources/data/```. The path to the desired data file can be specified in the ```RiskScoreICU``` class.

### Model

The trained XGBoost model is located in ```src/main/resources/models/```. The model has been trained on ICU data collected up until May 2021.   
Measurements required by the model with corresponding LOINC codes and aggregation methods are defined in ```src/main/resources/data/model_features.json```.

### Prediction

To obtain patient-wise risk scores, compile and run the code with ```sh build.sh```.
Predictions are saved in ```src/main/resources/results/```.  

The option to retain the patient order in the result file can be revoked by setting ```retain_order=false``` in ```RiskScoreICU.java``` to add a layer of anonymization.

## Feature Processing

The model assumes __Urine output__ to be normalized by the patient's body weight (ml/kg). To this end, body weight is required to be provided as input, even though it is not consumed directly by the model. Normalization is carried out in the RiskScoreICU module. If no body weight is provided, urine output cannot be considered and will be treated as missing.