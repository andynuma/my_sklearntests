from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression


#aclImdbからデータを取ってくる
reviews_train = load_files("aclImdb/train/")
#トレーニングデータセット
text_train,y_train = reviews_train.data, reviews_train.target

#改行文字削除
text_train = [doc.replace(b"<br />",b" ") for doc in text_train]

#テストデータセット
reviews_test = load_files("aclImdb/test")
text_test,y_test = reviews_test.data, reviews_test.target

#tfidfを導入
pipe = make_pipeline(TfidfVectorizer(min_df=5,norm=None),LogisticRegression())
param_grid = {"logisticregression__C":[0.01,0.1,1,10]}

#グリッドサーチ
grid = GridSearchCV(pipe,param_grid,cv=5)
grid.fit(text_train,y_train)
print(grid.best_score_)

###tfidfの結果をみる
#グリッドサーチから求めた最良のモデルをvectorizerに格納
vectorizer = grid.best_estimator_.named_steps["tfidfvectorizer"]
#それを用いて訓練データを変換
X_train = vectorizer.transform(text_train)
#それぞれの特徴量のデータセット中での最大値を見つける
max_value = X_train.max(axis=0).toarray().ravel()
#インデックスを格納
sorted_byidf = max_value.argsort()

#特徴量名を取得
feature_names = np.array(vectorizer.get_feature_names())
#tfidfの低い特徴量
print(feature_names[sorted_byidf[:20]])
#tfidfの高い特徴量
print(feature_names[sorted_byidf[-20:]])
