# NPCI MLOps Assignment - 10
## Pipeline for Customer Churn data using Kubeflow

##  Project Overview  
This assignment is to implement  **pipeline using Kubeflow** to process **Customer Churn data** and train a machine learning model 

The pipeline includes the flowing components:  
- **Data Loading**: loading raw data.  
- **train_test split**: splitting data into train and test sets
- **Model Training**: Training a classification model.
- **Model Evaluation**: getting the model's performance metrics.

#### Column Description:
The dataset you'll be working with is a customer dataset from a **Credit Card company**, which includes the following features:


- **RowNumber:** corresponds to the record (row) number and has no effect on the output.
- **CustomerId:** contains random values and has no effect on customer leaving the bank.
- **Surname:** the surname of a customer has no impact on their decision to leave the bank.
- **CreditScore:** can have an effect on customer churn, since a customer with a higher credit score is less likely to leave the bank.
- **Geography:** A customer’s location can affect their decision to leave the bank.
- **Gender:** It’s interesting to explore whether gender plays a role in a customer leaving the bank.
- **Age:** This is certainly relevant since older customers are less likely to leave their bank than younger ones.
- **Tenure:** refers to the number of years that the customer has been a client of the bank. Normally, older clients are more loyal and less likely to leave a bank.
- **Balance:** is also a very good indicator of customer churn, as people with a higher balance in their accounts are less likely to leave the bank compared to those with lower balances.
- **NumOfProducts:** refers to the number of products that a customer has purchased through the bank.
- **HasCrCard:** denotes whether or not a customer has a credit card. This column is also relevant since people with credit card are less likely to leave the bank.
- **IsActiveMember:** Active customers are less likely to leave the bank.
- **EstimatedSalary:** as with balance, people with lower salaries are more likely to leave the bank compared to those with higher salaries.
- **Exited:** whether or not the customer left the bank. (0=No,1=Yes)

---
 
## Assignment Tasks.

### Note: Please refer to the lab guide material on Kubeflow, for all the commands needed to complete the following tasks.

**1. Understanding the given files**

You are provided with the following:
* **Dataset (`data/customer_churn.csv`)**
* **Pipeline script (`pipeline.py`)**


**2. Create a codespace**
* Create a codespace using the repository with default selection for Branch and Region, for Machine type select 4-core.

**3. Setting up Kubernetes cluster**
* Set up a single-node Minikube on your Codespace.
* Switch to root and start Minikube.
* Install kubectl to communicate with the Minikube cluster.


**4. Installing Kubeflow Pipelines**
* Install Kubeflow pipelines using the manifest files from the Git repository.
* Verify pod creation on the cluster and accessibility to the Kubeflow pipeline dashboard by port-forwarding

**5. Creating a Kubeflow Pipeline for an ML Application Python script and running it**
* pipeline.py script contains components and pipelines authored using the KFP Python SDK.
* Execute the Python file to compile pipelines to an intermediate representation YAML, and submit the
pipeline to run.
* Upload the YAML file to the Kubeflow UI.
* Execute it by clicking on Create Run. A window will appear to add Run details.
* For the Experiment field, choose the Default experiment and for the Run Type field, choose One-off to execute the pipeline only once.
* Under the Run parameters, you can change the values if you want. Then click on Start.
* Pipeline execution will start. You should be able to see the run execution


## Submission Guidelines
After completing the assignment by running the pipeline successfully, submit screenshots of your executions and commands. then,

  - Stage your changes and commit the files:
    ```
    git add .
    git commit -m "challenge Completed "
    ```
  - Push your changes to the GitHub repository:
    ```
    git push
    ```

Good luck, and happy coding!
