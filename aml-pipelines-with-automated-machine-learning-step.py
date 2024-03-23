#!/usr/bin/env python
# coding: utf-8

# Copyright (c) Microsoft Corporation. All rights reserved.  
# Licensed under the MIT License.

# ![Impressions](https://PixelServer20190423114238.azurewebsites.net/api/impressions/NotebookVM/how-to-use-azureml/machine-learning-pipelines/intro-to-pipelines/aml-pipelines-with-automated-machine-learning-step.png)

# # Azure Machine Learning Pipeline with AutoMLStep (Udacity Course 2)
# This notebook demonstrates the use of AutoMLStep in Azure Machine Learning Pipeline.

# ## Introduction
# In this example we showcase how you can use AzureML Dataset to load data for AutoML via AML Pipeline. 
# 
# If you are using an Azure Machine Learning Notebook VM, you are all set. Otherwise, make sure you have executed the [configuration](https://aka.ms/pl-config) before running this notebook.
# 
# In this notebook you will learn how to:
# 1. Create an `Experiment` in an existing `Workspace`.
# 2. Create or Attach existing AmlCompute to a workspace.
# 3. Define data loading in a `TabularDataset`.
# 4. Configure AutoML using `AutoMLConfig`.
# 5. Use AutoMLStep
# 6. Train the model using AmlCompute
# 7. Explore the results.
# 8. Test the best fitted model.

# ## Azure Machine Learning and Pipeline SDK-specific imports

# In[1]:


import logging
import os
import csv

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
import pkg_resources

import azureml.core
from azureml.core.experiment import Experiment
from azureml.core.workspace import Workspace
from azureml.train.automl import AutoMLConfig
from azureml.core.dataset import Dataset

from azureml.pipeline.steps import AutoMLStep

# Check core SDK version number
print("SDK version:", azureml.core.VERSION)


# ## Initialize Workspace
# Initialize a workspace object from persisted configuration. Make sure the config file is present at .\config.json

# In[2]:


ws = Workspace.from_config()
print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep = '\n')


# ## Create an Azure ML experiment
# Let's create an experiment named "automlstep-classification" and a folder to hold the training scripts. The script runs will be recorded under the experiment in Azure.
# 
# The best practice is to use separate folders for scripts and its dependent files for each step and specify that folder as the `source_directory` for the step. This helps reduce the size of the snapshot created for the step (only the specific folder is snapshotted). Since changes in any files in the `source_directory` would trigger a re-upload of the snapshot, this helps keep the reuse of the step when there are no changes in the `source_directory` of the step.
# 
# *Udacity Note:* There is no need to create an Azure ML experiment, this needs to re-use the experiment that was already created
# 

# In[3]:


# Choose a name for the run history container in the workspace.
# NOTE: update these to match your existing experiment name
experiment_name = 'AutoML_Proj2_Experiment'
project_folder = './Project2'

experiment = Experiment(ws, experiment_name)
experiment


# ### Create or Attach an AmlCompute cluster
# You will need to create a [compute target](https://docs.microsoft.com/azure/machine-learning/service/concept-azure-machine-learning-architecture#compute-target) for your AutoML run. In this tutorial, you get the default `AmlCompute` as your training compute resource.
# 
# **Udacity Note** There is no need to create a new compute target, it can re-use the previous cluster

# In[4]:


from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget
from azureml.core.compute_target import ComputeTargetException

# NOTE: update the cluster name to match the existing cluster
# Choose a name for your CPU cluster
amlcompute_cluster_name = "vm-cluster-proj2"

# Verify that cluster does not exist already
try:
    compute_target = ComputeTarget(workspace=ws, name=amlcompute_cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_D2_V2',# for GPU, use "STANDARD_NC6"
                                                           #vm_priority = 'lowpriority', # optional
                                                           max_nodes=4)
    compute_target = ComputeTarget.create(ws, amlcompute_cluster_name, compute_config)

compute_target.wait_for_completion(show_output=True, min_node_count = 1, timeout_in_minutes = 10)
# For a more detailed view of current AmlCompute status, use get_status().


# ## Data
# 
# **Udacity note:** Make sure the `key` is the same name as the dataset that is uploaded, and that the description matches. If it is hard to find or unknown, loop over the `ws.datasets.keys()` and `print()` them.
# If it *isn't* found because it was deleted, it can be recreated with the link that has the CSV 

# In[5]:


# Try to load the dataset from the Workspace. Otherwise, create it from the file
# NOTE: update the key to match the dataset name
found = False
key = "bankmarketing_proj2"
description_text = "Bank Marketing DataSet for Udacity Course 2"

if key in ws.datasets.keys(): 
        found = True
        dataset = ws.datasets[key] 

if not found:
        # Create AML Dataset and register it into Workspace
        example_data = 'https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv'
        dataset = Dataset.Tabular.from_delimited_files(example_data)        
        #Register Dataset in Workspace
        dataset = dataset.register(workspace=ws,
                                   name=key,
                                   description=description_text)


df = dataset.to_pandas_dataframe()
df.describe()


# ### Review the Dataset Result
# 
# You can peek the result of a TabularDataset at any range using `skip(i)` and `take(j).to_pandas_dataframe()`. Doing so evaluates only `j` records for all the steps in the TabularDataset, which makes it fast even against large datasets.
# 
# `TabularDataset` objects are composed of a list of transformation steps (optional).

# In[6]:


dataset.take(5).to_pandas_dataframe()


# ## Train
# This creates a general AutoML settings object.
# **Udacity notes:** These inputs must match what was used when training in the portal. `label_column_name` has to be `y` for example.

# In[7]:


automl_settings = {
    "experiment_timeout_minutes": 20,
    "max_concurrent_iterations": 5,
    "primary_metric" : 'AUC_weighted'
}
automl_config = AutoMLConfig(compute_target=compute_target,
                             task = "classification",
                             training_data=dataset,
                             label_column_name="y",   
                             path = project_folder,
                             enable_early_stopping= True,
                             featurization= 'auto',
                             debug_log = "automl_errors.log",
                             **automl_settings
                            )


# #### Create Pipeline and AutoMLStep
# 
# You can define outputs for the AutoMLStep using TrainingOutput.

# In[8]:


from azureml.pipeline.core import PipelineData, TrainingOutput

ds = ws.get_default_datastore()
metrics_output_name = 'metrics_output'
best_model_output_name = 'best_model_output'

metrics_data = PipelineData(name='metrics_data',
                           datastore=ds,
                           pipeline_output_name=metrics_output_name,
                           training_output=TrainingOutput(type='Metrics'))
model_data = PipelineData(name='model_data',
                           datastore=ds,
                           pipeline_output_name=best_model_output_name,
                           training_output=TrainingOutput(type='Model'))


# Create an AutoMLStep.

# In[9]:


automl_step = AutoMLStep(
    name='automl_module',
    automl_config=automl_config,
    outputs=[metrics_data, model_data],
    allow_reuse=True)


# In[10]:


from azureml.pipeline.core import Pipeline
pipeline = Pipeline(
    description="pipeline_with_automlstep",
    workspace=ws,    
    steps=[automl_step])


# In[11]:


pipeline_run = experiment.submit(pipeline)


# In[12]:


from azureml.widgets import RunDetails
RunDetails(pipeline_run).show()


# In[13]:


pipeline_run.wait_for_completion()


# ## Examine Results
# 
# ### Retrieve the metrics of all child runs
# Outputs of above run can be used as inputs of other steps in pipeline. In this tutorial, we will examine the outputs by retrieve output data and running some tests.

# In[14]:


metrics_output = pipeline_run.get_pipeline_output(metrics_output_name)
num_file_downloaded = metrics_output.download('.', show_progress=True)


# In[15]:


import json
with open(metrics_output._path_on_datastore) as f:
    metrics_output_result = f.read()
    
deserialized_metrics_output = json.loads(metrics_output_result)
df = pd.DataFrame(deserialized_metrics_output)
df


# ### Retrieve the Best Model

# In[16]:


# Retrieve best model from Pipeline Run
best_model_output = pipeline_run.get_pipeline_output(best_model_output_name)
num_file_downloaded = best_model_output.download('.', show_progress=True)


# In[17]:


import pickle

with open(best_model_output._path_on_datastore, "rb" ) as f:
    best_model = pickle.load(f)
best_model


# In[18]:


best_model.steps


# ### Test the Model
# #### Load Test Data
# For the test data, it should have the same preparation step as the train data. Otherwise it might get failed at the preprocessing step.

# In[19]:


dataset_test = Dataset.Tabular.from_delimited_files(path='https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_test.csv')
df_test = dataset_test.to_pandas_dataframe()
df_test = df_test[pd.notnull(df_test['y'])]

y_test = df_test['y']
X_test = df_test.drop(['y'], axis=1)


# #### Testing Our Best Fitted Model
# 
# We will use confusion matrix to see how our model works.

# In[20]:


from sklearn.metrics import confusion_matrix
ypred = best_model.predict(X_test)
cm = confusion_matrix(y_test, ypred)


# In[ ]:


# Visualize the confusion matrix
pd.DataFrame(cm).style.background_gradient(cmap='Blues', low=0, high=0.9)


# ## Publish and run from REST endpoint
# 
# Run the following code to publish the pipeline to your workspace. In your workspace in the portal, you can see metadata for the pipeline including run history and durations. You can also run the pipeline manually from the portal.
# 
# Additionally, publishing the pipeline enables a REST endpoint to rerun the pipeline from any HTTP library on any platform.
# 

# In[25]:


published_pipeline = pipeline_run.publish_pipeline(
    name="Bankmarketing Train", description="Training bankmarketing pipeline", version="1.0")

published_pipeline


# Authenticate once again, to retrieve the `auth_header` so that the endpoint can be used

# In[26]:


from azureml.core.authentication import InteractiveLoginAuthentication

interactive_auth = InteractiveLoginAuthentication()
auth_header = interactive_auth.get_authentication_header()



# Get the REST url from the endpoint property of the published pipeline object. You can also find the REST url in your workspace in the portal. Build an HTTP POST request to the endpoint, specifying your authentication header. Additionally, add a JSON payload object with the experiment name and the batch size parameter. As a reminder, the process_count_per_node is passed through to ParallelRunStep because you defined it is defined as a PipelineParameter object in the step configuration.
# 
# Make the request to trigger the run. Access the Id key from the response dict to get the value of the run id.
# 

# In[27]:


import requests

rest_endpoint = published_pipeline.endpoint
response = requests.post(rest_endpoint, 
                         headers=auth_header, 
                         json={"ExperimentName": "pipeline-rest-endpoint"}
                        )


# In[28]:


try:
    response.raise_for_status()
except Exception:    
    raise Exception("Received bad response from the endpoint: {}\n"
                    "Response Code: {}\n"
                    "Headers: {}\n"
                    "Content: {}".format(rest_endpoint, response.status_code, response.headers, response.content))

run_id = response.json().get('Id')
print('Submitted pipeline run: ', run_id)


# Use the run id to monitor the status of the new run. This will take another 10-15 min to run and will look similar to the previous pipeline run, so if you don't need to see another pipeline run, you can skip watching the full output.

# In[29]:


from azureml.pipeline.core.run import PipelineRun
from azureml.widgets import RunDetails

published_pipeline_run = PipelineRun(ws.experiments["pipeline-rest-endpoint"], run_id)
RunDetails(published_pipeline_run).show()


# In[ ]:




