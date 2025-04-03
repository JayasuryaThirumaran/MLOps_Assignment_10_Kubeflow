from kfp.dsl import component, pipeline, Input, Output, Dataset, Model
import kfp

# Pipeline Component-1: Data Ingestion
@component(
    packages_to_install=["pandas", "numpy", "scikit-learn"],
    base_image="python:3.10-slim",
)
def load_churn_data(drop_missing_vals: bool, churn_dataset: Output[Dataset]):
    import pandas as pd
    # Load Customer Churn dataset
    df = pd.read_csv('data/customer_churn.csv')

    if drop_missing_vals:
        df = df.dropna()

    with open(customer_churn_dataset.path, 'w') as file:
        df.to_csv(file, index=False)


# Pipeline Component-2: Train-Test Split
@component(
    packages_to_install=["pandas", "numpy", "scikit-learn"],
    base_image="python:3.10-slim",
)
def train_test_split_churn(
    input_churn_dataset: Input[Dataset],
    X_train: Output[Dataset],
    X_test: Output[Dataset],
    y_train: Output[Dataset],
    y_test: Output[Dataset],
    test_size: float,
    random_state: int,
):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(input_churn_dataset.path)
    X = df.drop(['Exited'], axis=1)
    y = df[['Exited']]

    X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(X, y, test_size=test_size, random_state=random_state)

    X_train_data.to_csv(X_train.path, index=False)
    X_test_data.to_csv(X_test.path, index=False)
    y_train_data.to_csv(y_train.path, index=False)
    y_test_data.to_csv(y_test.path, index=False)


# Pipeline Component-3: Model Training
@component(
    packages_to_install=["pandas", "numpy", "scikit-learn"],
    base_image="python:3.10-slim",
)
def train_churn_model(
    X_train: Input[Dataset],
    y_train: Input[Dataset],
    model_output: Output[Model],
    n_estimators: int,
    random_state: int,
):
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    import pickle

    X_train_data = pd.read_csv(X_train.path)
    y_train_data = pd.read_csv(y_train.path)

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train_data, y_train_data['Exited'].values)

    with open(model_output.path, 'wb') as file:
        pickle.dump(model, file)


# Pipeline Component-4: Model Evaluation
@component(
    packages_to_install=["pandas", "numpy", "scikit-learn"],
    base_image="python:3.10-slim",
)
def evaluate_churn_model(
    X_test: Input[Dataset],
    y_test: Input[Dataset],
    model_path: Input[Model],
    metrics_output: Output[Dataset]
):
    import pandas as pd
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import pickle

    X_test_data = pd.read_csv(X_test.path)
    y_test_data = pd.read_csv(y_test.path)

    with open(model_path.path, 'rb') as file:
        model = pickle.load(file)

    y_pred = model.predict(X_test_data)
    metrics = {
        'Accuracy': accuracy_score(y_test_data, y_pred),
        'Precision': precision_score(y_test_data, y_pred, average='weighted'),
        'Recall': recall_score(y_test_data, y_pred, average='weighted'),
        'F1-Score': f1_score(y_test_data, y_pred, average='weighted')
    }
    pd.DataFrame([metrics]).to_csv(metrics_output.path, index=False)


# Pipeline Definition
@pipeline(name="customer-churn-pipeline", description="Pipeline for Customer Churn Prediction")
def customer_churn_pipeline(
    drop_missing_vals: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 100,
):
    data_task = load_churn_data(drop_missing_vals=drop_missing_vals)
    split_task = train_test_split_churn(
        input_churn_dataset=data_task.outputs['customer_churn_dataset'],
        test_size=test_size,
        random_state=random_state)
    model_task = train_churn_model(
        X_train=split_task.outputs['X_train'],
        y_train=split_task.outputs['y_train'],
        n_estimators=n_estimators,
        random_state=random_state)
    eval_task = evaluate_churn_model(
        X_test=split_task.outputs['X_test'],
        y_test=split_task.outputs['y_test'],
        model_path=model_task.outputs['model_output'])


# Compile Pipeline
if __name__ == '__main__':
    kfp.compiler.Compiler().compile(customer_churn_pipeline, 'customer_churn_pipeline.yaml')
