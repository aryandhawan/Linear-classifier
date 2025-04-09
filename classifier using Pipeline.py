from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

data=load_breast_cancer()
x=data.data
y=data.target
param_grid = {
    'classification__C': [0.01, 0.1, 0.5, 1.0],
    'classification__penalty': ['l1', 'l2'],
    'classification__solver': ['liblinear']
}

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

pipe=Pipeline([
    ('scaler',StandardScaler()),
    ('classification',LogisticRegression(max_iter=1000))])

grid=GridSearchCV(pipe,param_grid=param_grid,cv=5,scoring='accuracy')

grid.fit(x_train,y_train)

best_model=grid.best_estimator_

y_pred=best_model.predict(x_test)
print(f'Accuracy Score:{accuracy_score(y_true=y_test,y_pred=y_pred)}')
print(f'Classification report: {classification_report(y_true=y_test,y_pred=y_pred)}')

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot it
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=data.target_names, yticklabels=data.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')  # Save it to include in GitHub
plt.show()
