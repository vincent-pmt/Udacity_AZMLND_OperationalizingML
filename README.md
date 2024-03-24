<!-- #region -->
# Udacity Project: AML - Operationalizing Machine Learning

Utilize the Bank Marketing dataset to establish a machine learning model in Azure Machine Learning, focuses on enhancing the bank's marketing strategies. The model was developed with the assistance of Azure's AutoML capabilities, utilization of the best model are facilitated through a REST endpoint. 
Additionally, utilize a pipeline for the scalable and efficient way to build, optimize, and manage machine learning workflows
scalable and efficient way to build, optimize, and manage machine learning workflows, publish it for wider access.

## Architectural Diagram

![Architectural Diagram](/images/ArchitectualDiagram.jpg)

## Key Steps
#### Step 1: Authentication
- Install the Azure Machine Learning Extension
- Create Service Principal

![Create SP](/images/Authentication_CreateSP.jpeg)

- Associate it with your specific workspace

![Assign Role](/images/Authentication_AssignRole.jpeg)

#### Step 2: Automated ML Experiment
- Create bankmarketing dataset

![Create dataset](/images/RegisteredDatasets.jpeg)

- Create an experiment using Automated ML

![Create an experiment](/images/ExperimentCompleted.jpeg)

- Show BestModel

![Best Model](/images/BestModel.jpeg)

#### Step 3: Deploy the Best Model

- Deploy best model 
- Enable authentication (key-based)
- Using Azure Container Instance
![Deploy best model](/images/BestModel_Endpoint.jpeg)

#### Step 4: Enable Logging
- Enable application insight (by calling service.update(enable_app_insights=True))

![Enable AI](/images/Logs_EnableAI.jpeg)

#### Step 4: Enable Logging
- Enable application insight (by calling service.update(enable_app_insights=True))

![App Insight](/images/Logs_AIEnabled.jpeg)

![Enable AI](/images/Logs_EnableAI.jpeg)

#### Step 5: Swagger Documentation
- Consume the deployed model using Swagger.

![Swagger](/images/swagger_localhost.jpeg)

#### Step 6: Consume Model Endpoints
- use the endpoint.py script provided to interact with the trained model

![consume Model Endpoint](/images/consumeModelEndpoint.jpeg)

#### Step 7: Create, Publish and Consume a Pipeline

- Create pipeline

![create pipeline](/images/pipeline_running.jpeg)

- Show pipeline endpoint, and active status

![pipeline endpoint](/images/pipeline_endpointactive.jpeg)

- pipeline - bankmarkerting datasets

![pipeline datasets](/images/pipeline_datasets.jpeg)

- Notebook - RunDetail Widget

![pipeline run details](/images/notebook_RunDetails.jpeg)

- Pipeline - Schedule Run

![pipeline schedule 1](/images/pipeline_schedule.jpeg)

- Pipeline - Schedule - Job details

![pipeline schedule 1](/images/pipeline_schedule2.jpeg)

## Screen Recording
The screencast should demonstrate:
- Deployed ML Model Endpoint
- Deployed Pipeline
- Available ML Model
- Endpoint Test

Please click here for viewing. (Please enable the caption on this video)

[![Watch the video](https://i9.ytimg.com/vi_webp/THMG2i8j_Cs/mq2.webp?sqp=CIyxgbAG-oaymwEmCMACELQB8quKqQMa8AEB-AH-CYAC0AWKAgwIABABGBMgTih_MA8=&rs=AOn4CLAKmwH7i8juRmatqFPVPQ3p6cuQxA)](https://youtu.be/THMG2i8j_Cs)

## Standout Suggestions
<!-- #endregion -->
