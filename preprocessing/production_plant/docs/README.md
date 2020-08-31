## Context

This dataset was used in previous research projects, for example in IMPROVE.

## Content

The use case focuses on the prediction of the condition of an important component within production lines. The condition of this component is important for the function of the plant and the resulting product quality. Data for 8 run-to-failure experiments were provided and 8 features related to the component were selected. Training and prediction data were selected using the leave-one-out method: data from the component under test were selected as the target for the prediction. A set amount of data of all other components were selected and combined to serve as training data for the 'new' condition. A SOM was trained on a training data to represent the 'new' condition. The degradation of the component under test was calculated and visualized. This procedure was repeated for all 8 data sets to get a prediction of the degradation for all components. The prediction worked for all cases which were labeled with a certain type of wear by experts. Furthermore, one of the components did not show signs of wear according to experts which was also confirmed by the model.

## Acknowledgements

This dataset is publicly available for anyone to use under the following terms.

von Birgelen, Alexander; Buratti, Davide; Mager, Jens; Niggemann, Oliver: Self-Organizing Maps for Anomaly Localization and Predictive Maintenance in Cyber-Physical Production Systems. In: 51st CIRP Conference on Manufacturing Systems (CIRP CMS 2018) CIRP-CMS, May 2018.

Paper available open access: https://authors.elsevier.com/sd/article/S221282711830307X

IMPROVE has received funding from the European Union's Horizon 2020 research and innovation programme under Grant Agreement No. 678867