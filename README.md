# Deploying a Titanic Survival Prediction Model with Docker and Flask

## Introduction
This project demonstrates a practical approach to deploying machine learning models using Docker and Flask, followed by an exploration into using scikit-learn's Pipeline feature for streamlining machine learning workflows. It's structured into two key sections: The Basic and The Intermediate, catering to different levels of expertise and application complexity.

## Section 1: The Basic

### Overview
In this section, we delve into the fundamentals of deploying a machine learning model. The process is broken down into two primary steps: containerizing the model with Docker for easy deployment and management, and exposing the model as an API using Flask. This enables easy integration of the model into various applications or services.

### Topics Covered
1. **Model Deployment in Docker**: Learn how to encapsulate your machine learning model and its environment within a Docker container. This step ensures consistency across various deployment environments and simplifies deployment processes.

2. **API Exposition with Flask**: Discover how to create a simple web server using Flask that serves your machine learning model. We'll illustrate how to receive data through API calls, process it, and return predictions.

For this section, please refer to [Jupyter notebook](Deploying%20a%20Titanic%20Survival%20Prediction%20Model%20with%20Docker%20and%20Flask.ipynb) in this repository.

## Section 2: The Intermediate

### Overview
Building upon the basics, this section focuses on the advanced implementation of machine learning workflows using **scikit-learn's Pipeline feature**. Pipelines streamline the process of chaining multiple steps such as data preprocessing, feature selection, and model training into a single, reusable workflow.

### Topics Covered

1. [build-titanic-model](build-titanic-model.py): This script develops a logistic regression model to predict the likelihood of survival on the Titanic. It incorporates custom data transformers for feature processing, followed by training and evaluating the logistic regression model. The evaluation metrics include accuracy, confusion matrix, and a classification report. Finally, the model, integrated with its data preprocessing pipeline, is saved, demonstrating a complete workflow from data preparation to model deployment.

2. [hypertune-titanic-model](hypertune-titanic-model.py): The script focuses on optimizing a logistic regression model for Titanic survival prediction using hyperparameter tuning. It employs a randomized search over a defined parameter space to identify the best model configurations. Key aspects include custom data preprocessing, hyperparameter search for logistic regression, and saving the optimally tuned model. This highlights the importance of fine-tuning in enhancing model performance and accuracy.

3. [app.py](app.py): This script creates a Flask web application to serve predictions from the Titanic survival model. It loads a pre-trained model, sets up an endpoint for prediction requests, and processes incoming JSON data to return survival predictions. The application is equipped with logging to record prediction details and errors, enhancing monitoring and debugging capabilities. The structure demonstrates how to integrate a machine learning model into a web service for real-time predictions.

For full turorial on scikit-learn Pipeline, please refer to this [repository](https://github.com/swatakit/ml-workflow-sklearn-pipeline)