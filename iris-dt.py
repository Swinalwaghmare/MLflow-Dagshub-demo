import mlflow
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import dagshub
dagshub.init(repo_owner='Swinalwaghmare', repo_name='MLflow-Dagshub-demo', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/Swinalwaghmare/MLflow-Dagshub-demo.mlflow")

# Load the iris dataset
iris = load_iris()
X = iris.data 
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

max_depth = 20

# apply mlflow
mlflow.set_experiment('iris-dt')

#context manager
with mlflow.start_run():

    dt = DecisionTreeClassifier(max_depth=max_depth)
    dt.fit(X_train, y_train)

    y_pred = dt.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    # confusion-mtx = confusion_matrix(y_test, y_pred)

    mlflow.log_metric('accuracy',accuracy)

    mlflow.log_param('max_depth',max_depth)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    plt.savefig('confusion_matrix.png')
    mlflow.log_artifact("confusion_matrix.png")

    mlflow.log_artifact(__file__)

    mlflow.sklearn.log_model('dt', "decision tree")

    mlflow.set_tag("author",'swinal')
    mlflow.set_tag("model",'decision tree')
    
    print(f"accuracy:{accuracy}")
