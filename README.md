# 鸢尾花分类（Iris Classification）

基于 UCI 鸢尾花数据集的机器学习分类项目，使用 scikit-learn 构建并对比了多种分类模型。

## 📌 项目简介

鸢尾花数据集是机器学习中最经典的入门数据集之一。本项目通过完整的建模流程，对三种鸢尾花（setosa、versicolor、virginica）进行分类，并对比不同模型的性能。

## 📁 项目结构
ml-notes/
└── iris-classification/
├── iris classification.py # 主脚本
├── README.md # 项目说明
└── .gitignore # Git 忽略规则

## 🔧 环境依赖

- Python 3.9+
- scikit-learn
- pandas
- numpy
- ucimlrepo

安装依赖：

bash
pip install scikit-learn pandas numpy ucimlrepo
数据通过 ucimlrepo 从 UCI 机器学习库在线获取：
from ucimlrepo import fetch_ucirepo

iris = fetch_ucirepo(id=53)
X = iris.data.features
y = iris.data.targets.values.ravel()
🚀 建模流程
数据加载：从 UCI 获取数据

数据划分：按 8:2 划分训练集和测试集，分层抽样

建立 Pipeline：标准化 + 分类器

模型训练与评估：准确率、分类报告、混淆矩阵

交叉验证：5 折交叉验证

模型对比：对比多个分类器

新样本预测：用最优模型预测新数据
🤖 对比的模型
模型	5折交叉验证平均准确率
LogisticRegression	0.960 (±0.039)
KNN	0.953 (±0.034)
SVM	0.967 (±0.021)
DecisionTree	0.953 (±0.034)
RandomForest	0.967 (±0.021)
📈 结果
单次划分测试集结果（30 个样本）

准确率：0.9333

混淆矩阵：

text
[[10  0  0]
 [ 0  9  1]
 [ 0  1  9]]
分类报告

类别	precision	recall	f1-score	support
Iris-setosa	1.00	1.00	1.00	10
Iris-versicolor	0.90	0.90	0.90	10
Iris-virginica	0.90	0.90	0.90	10
accuracy			0.93	30
macro avg	0.93	0.93	0.93	30
weighted avg	0.93	0.93	0.93	30
SVM 和 RandomForest 表现最好，交叉验证平均准确率均为 0.967。
新样本预测：
import pandas as pd

new_sample = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=X.columns
)
pred = best_pipe.predict(new_sample)
proba = best_pipe.predict_proba(new_sample)

target_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
print("预测类别:", target_names[pred[0]])
print("各类别概率:", proba[0])
