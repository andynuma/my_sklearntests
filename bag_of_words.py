"""
IMDbのレビューをBoWに変換しそのあと定量的にクラス分類器を作成し評価するプログラム
（注意）クラス分類器はネガポジの分類とは関係がない
最初はそのままBoWにしている
そのあとは5つ以上の文書に含まれる単語のみを選んでBoWに
"""
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


#aclImdbからデータを取ってくる
reviews_train = load_files("aclImdb/train/")
#トレーニングデータセット
text_train,y_train = reviews_train.data, reviews_train.target

#改行文字削除
text_train = [doc.replace(b"<br />",b" ") for doc in text_train]

#テストデータセット
reviews_test = load_files("aclImdb/test")
text_test,y_test = reviews_test.data, reviews_test.target

#BoWここから
#BoWのクラスを生成
vect = CountVectorizer()

#fit(トークン分割と、ボキャブラリ構築（番号付け）)
vect.fit(text_train)

#エンコード（疎行列）、BoW表現のX_trainを作る
X_train = vect.transform(text_train)

#shape表示
print("X_train(before):",repr(X_train))

###ここまででBoWが完成####

#特徴量を見てみる,前処理してないので多いかも
feature_names = vect.get_feature_names()
print("featurs number(before):",len(feature_names))
#print(feature_names[::2000])

#交差検証のためにロジスティック回帰を使う,高次元の疎なデータに使えるらしい...
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

#交差検証、cvは分割数
scores = cross_val_score(LogisticRegression(),X_train,y_train,cv=5)

#平均交差検証スコア
print("score(before)",np.mean(scores))

#グリッドサーチ
from sklearn.model_selection import GridSearchCV

###ロジスティック回帰の正則化パラメータCを交差検証でチューニング
#与えるCを適当に決める
param_grid = {"C":[0.001,0.01,0.1,1,10]}
#チューニング
grid = GridSearchCV(LogisticRegression(),param_grid,cv=5)
#fit
grid.fit(X_train,y_train)
#ベストスコア
print("best_score(before)",grid.best_score_)
#最良パラメータ
print("best_params(before)",grid.best_params_)



####テストセットから汎化性能を見てみる
X_test = vect.transform(text_test)
print("test_score:",grid.score(X_test,y_test))

#5つ以上の文章に登場しているトークンだけを用いるようにする
#min_dfで設定
vect = CountVectorizer(min_df=5).fit(text_train)
X_train = vect.transform(text_train)

#shape表示
print("X_train:",repr(X_train))

#特徴量をみてみる
feature_names = vect.get_feature_names()
print("features number :",len(feature_names))
#print(feature_names[::2000])

#再びグリッドサーチ
grid = GridSearchCV(LogisticRegression(),param_grid,cv=5)
grid.fit(X_train,y_train)
#ベストスコア
print("best_score:",grid.best_score_)
#最良パラメータ
print("best_params:",grid.best_params_)
