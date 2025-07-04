# Databricks notebook source
# MAGIC %md
# MAGIC <img src="https://mlflow.org/docs/latest/images/logo-light.svg" width="240px" alt="MLflow logo" />
# MAGIC
# MAGIC # MLflow Tracing with OpenAI
# MAGIC
# MAGIC This notebook demonstrates how to use MLflow's OpenAI tracing integration to capture and analyze your GenAI application's execution.
# MAGIC
# MAGIC > **Learn more** about this integration in the <a href="https://mlflow.org/docs/latest/tracing/integrations/openai" target="_blank">MLflow OpenAI tracing documentation</a>.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Installation
# MAGIC
# MAGIC First, let's install the required packages and restart the Python kernel:

# COMMAND ----------

# MAGIC %pip install -U mlflow openai
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Enable Autologging (if supported)
# MAGIC
# MAGIC Now, we'll enable MLflow's automatic tracing functionality, if supported. This captures inputs, outputs, and performance metrics without changing your application code:

# COMMAND ----------

# Enable MLflow autologging for OpenAI
import mlflow

mlflow.openai.autolog()
mlflow.set_experiment(experiment_id=253708371710781)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Run Your Application
# MAGIC
# MAGIC With autologging enabled, we can now run our application code. MLflow will automatically track all OpenAI operations:
# MAGIC
# MAGIC > **Tip**: After running the code below, you'll see traces in the MLflow UI and directly below the cell output!

# COMMAND ----------

# Example application code
from openai import OpenAI

# Ensure that the OPENAI_API_KEY environment variable is set
client = OpenAI()

messages = [
  {
    "role": "system", 
    "content": "You are a helpful assistant."
  },
  {
    "role": "user",
    "content": "Hello!"
  }
]

client.chat.completions.create(model="gpt-4o-mini", messages=messages)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: View and Analyze Traces
# MAGIC
# MAGIC After running your code, MLflow captures detailed traces of each step. You can:
# MAGIC
# MAGIC * View traces directly below the cell output
# MAGIC * Explore the full trace details in the MLflow UI
# MAGIC * Compare multiple runs to analyze performance and outputs
# MAGIC
# MAGIC Try modifying the input parameters above and run the cell again to see how the traces compare!

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC * Experiment with different inputs to see how they affect the outputs
# MAGIC * Add more complex OpenAI workflows to your application
# MAGIC * Check out other MLflow tracing integrations for your GenAI stack
