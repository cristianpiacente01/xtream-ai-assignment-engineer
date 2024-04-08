# Diamond price prediction architecture on Google Cloud Platform - by Cristian Piacente

## Introduction

To upgrade the training and serving infrastructure, we're going to use Google Cloud Platform as the cloud provider: in particular, we're exploiting the services offered by **Google Cloud's Vertex AI**.

GCP's Vertex AI is a seamless suite of ML tools that offer high computational power and flexibility, needed to make our model reach other businesses.

## Data Engineering

First of all, we need to handle heavy workloads, quick transformations and to have an efficient data storage. We need powerful tools that provide scalability and cost-effectiveness.

### Dataflow

Dataflow will handle the data processing, such as complex **ETL operations**.  It's scalable and efficient for processing data in real-time or batch: we don't have to worry about the volume or velocity of the data.

### BigQuery

BigQuery is Google's serverless, highly scalable and cost-effective multi-cloud **data warehouse**. It is used for efficiently storing and querying large-scale data.

## Data Analysis

Data Analysis is necessary before performing feature engineering and model training. We need a shared space for deep analysis.

### Vertex AI Workbench

Vertex AI Workbench offers a collaborative **Jupyter notebook** environment fully integrated with other GCP services, including BigQuery. We can explore and visualize the data, as well as experiment and begin model training.

## Feature Engineering

Feature Engineering improves model performance. We need a centralized feature store to keep consistency across ML pipelines.

### Vertex AI Feature Store

Vertex AI Feature Store is a **unified store for** serving, sharing and re-using **ML features**. It reduces redundancy and it ensures consistency between training and prediction.

## Model Training

We need to simplify this complex and time-consuming process, and we need to have an organized structure for model management.

### AutoML

AutoML allows us to build **models without writing code**, by automating the process of applying ML with high-quality results.

### Vertex AI Model Registry

Vertex AI Model Registry is a **central repository** used for managing the lifecycle of ML models. It supports versioning, metadata storage and model lineage.

## Model Deployment

To expose the model to other businesses, we need a scalable serving mechanism that abstracts away infrastructure concerns.

### Vertex AI Prediction

Vertex AI Prediction deploys the model as a **REST API** endpoint. It easily serves predictions with low latency and reliability.

## MLOps

Continuous monitoring and updating ML models are very important to maintain performance. We need to automatically detect anomalies, and we also need to automate deploying. 

### Vertex AI Model Monitoring

Vertex AI Model Monitoring **monitors model performance** to alert us about any anomalies or drift in the data (changes in patterns) that may affect predictions.

### Vertex AI Pipelines

Vertex AI Pipelines automates the **deployment, monitoring and governance**. It can be triggered by Vertex AI Model Monitoring: for instance, it's possible to re-train a model when an anomaly is detected. In this way, we can obtain a dynamic ML environment.

## Conclusion

Here is the new architecture overview:

 1. **Data ingestion and preprocessing**
	 - Dataflow
	 - BigQuery
 2. **Explanatory Data Analysis**
	 - Vertex AI Workbench
 3. **Feature engineering**
	 - Vertex AI Feature Store
 4. **Model training and management**
	 - AutoML
	 - Vertex AI Model Registry
 5. **Deployment**
 	 - Vertex AI Prediction
 6. **Continuous monitoring and updating**
  	 - Vertex AI Model Monitoring
  	 - Vertex AI Pipelines

Don Francesco can now make even more money! ;)