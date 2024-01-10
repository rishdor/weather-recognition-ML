# Weather Image Classification

## Description

This project uses a Convolutional Neural Network (CNN) to classify weather images into seven categories.

## Dataset

The model was trained on a combination of three different weather image datasets. All images were resized to 200x200 pixels for uniformity before training. There are 350 images for each class. The datasets used are:
- [Dataset 1](https://www.kaggle.com/datasets/jehanbhathena/weather-dataset)
- [Dataset 2](https://www.kaggle.com/datasets/polavr/twoclass-weather-classification)
- [Dataset 3](https://www.kaggle.com/datasets/vijaygiitk/multiclass-weather-dataset)

## Classes

The model classifies images into the following categories:
- `cloudy`
- `foggy`
- `lightning`
- `rainy`
- `snow`
- `sunny`
- `sunrise`

## Model Performance

The model's performance metrics are as follows:

``` plaintext
              precision    recall  f1-score   support

           0       0.77      0.69      0.73        39
           1       0.82      0.90      0.86        40
           2       0.80      0.93      0.86        30
           3       0.71      0.46      0.56        37
           4       0.62      0.82      0.71        34
           5       0.88      0.81      0.85        37
           6       0.96      0.96      0.96        28

    accuracy                           0.79       245
   macro avg       0.80      0.80      0.79       245
weighted avg       0.79      0.79      0.78       245
```
## Training and Validation Accuracy and Loss Graph

![training_validation](https://github.com/rishdor/weather-recognition-ML/assets/66086647/f7fbf25d-87de-43e0-ab6f-094f57effa84)
