from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(load_iris().data, load_iris().target)
print("Model trained with accuracy:", model.score(load_iris().data, load_iris().target))
