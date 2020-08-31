# Prognostics Benchmark Pre-Processing
Each folder contains all assets of the respective dataset and has the following structure:
```
/docs - contains original documentation
/raw - unpacked, but raw data*
/preprocessed - preprocessed data
preprocessing.py - script to preprocess the data
README.md - contains a short description, a download link and the citation 
```

| Dataset-ID       | Has Anomaly Labels? | # of channels** |    of which categorical    | Weak Supervision? | Number of Failures |
|------------------|-------------|------------------|----------------------------|-------------------|--------------------|
| harddrive        |     No      |       64         |            2               |        Yes        |          11.006    |
| production_plant |     No      |       26         |            0               |        Yes        |          8         |
| turbofan_engine  |     No      |       25         |            3               |        Yes        |          708*      |
| water_pump       |     No      |       52         |            0               |        Yes        |          7         |

\* Validation data which is cut off before the failure to predict the RUL is not included - only full run-to-failures are counted.

\** Including Timestamp