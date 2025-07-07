# Databricks notebook source
# MAGIC %md
# MAGIC # THE bulid up of confussion matrix

# COMMAND ----------

import mlflow
import mlflow.sklearn
# first we import important fils from sklearn


from sklearn.datasets import load_breast_cancer
# we are importing the 0 and 1 binary classfication
from sklearn.model_selection import train_test_split
# we are importing train test to split data
from sklearn.linear_model import LogisticRegression
# we are importing the logistic regression for algos
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
# in this we are impoting all the required mmeters for 
# confusion matrix to compute confusion matrix
# accuracy_score for accurate score
# precision score for percentage of correct positves
# recall score for the founded positives
# f1 score for the balance between precision and recall
# confusion matrix display to display the confusion matrix
import matplotlib.pyplot as plt
# we are importing matplotlib for visuals

# COMMAND ----------

data = load_breast_cancer()
# in this step we are loading the data form sklearn for our use

# COMMAND ----------

# Now we will split the data into train and test give x and y 
x = data.data
y = data.target
# now we split the data into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# COMMAND ----------

with mlflow.start_run():
    # now we create the model
    model = LogisticRegression()
    model.fit(x_train, y_train)
    # now we use the fit command for the model to train
    y_pred = model.predict(x_test)
    # in this command we are predicting the data using a prediction model
    # we have to use the confusion matrix command to display the prediction
    cm = confusion_matrix(y_test, y_pred)
    # we are using the confusion matrix command to display the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names)
    # we are using the confusion matrix display command to display the confusion matrix
    disp.plot()
    plt.title('Confusion Matrix')
    plt.show()

# COMMAND ----------

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
# In this command we are using the accuracy score command to display the accuracy score
print(f"Precision: {precision_score(y_test, y_pred):.2f}")
# In this command we are using the precision score command to display the precision score
print(f"Recall: {recall_score(y_test, y_pred):.2f}")
# In this command we are using the recall score command to display the recall score
print(f"F1 Score: {f1_score(y_test, y_pred):.2f}")
# In this command we are using the f1 score command to display the f1 score
