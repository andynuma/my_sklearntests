import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

#データセット
iris_dataset = load_iris()

#ほんとは可視化して機械学習できるかを考えること（pandasのDataFrameに入れて特徴量についてみる）

#訓練データとテストデータを分ける
#iris_datasetのdataには雄しべなどの数値データ、targetには0,1,2でirisの種類が表現されている(ラベル)
X_train,X_test ,y_train,y_test = train_test_split(iris_dataset["data"],iris_dataset["target"],random_state = 0)


#k近傍法 近傍点の数は1にしている
knn = KNeighborsClassifier(n_neighbors = 1)

#fit (データ、ラベル)
knn.fit(X_train,y_train)

#prediction
y_pred = knn.predict(X_test)

print(np.mean(y_test == y_pred))
print(knn.score(X_test,y_test))
