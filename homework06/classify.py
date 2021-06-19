from bayes import NaiveBayesClassifier
from db import News, session

X, y = [], []
mockup = NaiveBayesClassifier(alpha=0.05)
k = session()
rows = k.query(News).filter(News.label != None).all()
for row in rows:
    X.append(f"{row.title} {row.author} {row.url}")
    y.append(row.label)
X_train, y_train, X_test, y_test = X[:735], y[:735], X[735:], y[735:]
mockup.fit(X, y)
print(f"{mockup.score(X_test, y_test)}")