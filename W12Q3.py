from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC()
}

print("\nModel Comparison:\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)

    print("Model:", name)

    # Actual vs Predicted (first 5 rows)
    comp = pd.DataFrame({"Actual": y_test, "Predicted": pred})
    print(comp.head(5))

    print("Accuracy:", round(acc, 2))
    print("-" * 40)
