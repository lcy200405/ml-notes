from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
iris = fetch_ucirepo(id=53) 
  
# data (as pandas dataframes) 
X = iris.data.features 
y = iris.data.targets 
  
# metadata 
print(iris.metadata) 
  
# variable information 
print(iris.variables) 
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split

iris = fetch_ucirepo(id=53)
X = iris.data.features          # DataFrame
y = iris.data.targets.values.ravel()   # 转成一维数组，避免格式问题

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=200, random_state=42)
)

pipe.fit(X_train, y_train)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = pipe.predict(X_test)

print("准确率:", accuracy_score(y_test, y_pred))
print("\n分类报告:\n", classification_report(y_test, y_pred))
print("\n混淆矩阵:\n", confusion_matrix(y_test, y_pred))
from sklearn.model_selection import cross_val_score

scores = cross_val_score(pipe, X, y, cv=5, scoring='accuracy')
print("5折交叉验证:", scores)
print("平均准确率: %.3f ± %.3f" % (scores.mean(), scores.std()))
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

models = {
    'LogisticRegression': LogisticRegression(max_iter=200, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=3),
    'SVM': SVC(kernel='rbf', random_state=42),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'RandomForest': RandomForestClassifier(random_state=42)
}

for name, model in models.items():
    pipe = make_pipeline(StandardScaler(), model)
    scores = cross_val_score(pipe, X, y, cv=5, scoring='accuracy')
    print(f"{name:20s} 平均准确率: {scores.mean():.3f} (±{scores.std():.3f})")
best_pipe = make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True, random_state=42))
best_pipe.fit(X_train, y_train)

# 预测一个新样本（花萼长5.1，宽3.5，花瓣长1.4，宽0.2）
new_sample = [[5.1, 3.5, 1.4, 0.2]]
pred = best_pipe.predict(new_sample)
proba = best_pipe.predict_proba(new_sample)

print("预测类别:", pred[0])
print("各类别概率:", proba[0])
