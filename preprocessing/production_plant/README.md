## production_plant
8 run to failures of a production plant

```
@misc{ProductionPlantData,
  title = {Production {{Plant Data}} for {{Condition Monitoring}}},
  abstract = {Prediction of the condition of an important component},
  howpublished = {https://kaggle.com/inIT-OWL/production-plant-data-for-condition-monitoring},
  language = {en}
}
```

The exact timestamp of the error cant be provided. The 8 runs are showing a process from the starting condition (new components) till the components were changed (component was not working correctly anymore). So the failure must be in the last observations. You can take the last 500 observation if you want to be sure to have the failure.
We recommend to "learn" the first 20% of the observations (or build a model of the first 20%) with your machine learning software or algorithm. With that you will have the normal behavior of the plant. Depending on your learning algorithm it may take a few more observations. After you learned the normal behavior or your model, compare your result with the rest of the data or run an anomaly detection to find the point of the failure.

The runs for C7 and C13 are split into two files, as these runs have had a brief break. We have no information why. It could only be a short check up. The break was about 100 timestamps long.