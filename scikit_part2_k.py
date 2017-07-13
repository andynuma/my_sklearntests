from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

"""
breast_cancerのデータセットを用いて、k近傍法のkを
1-10にして見たときの、学習の精度を可視化する
"""

cancer = load_breast_cancer()
X_train,X_test ,y_train,y_test = train_test_split(cancer.data,cancer.target,random_state = 0,stratify=cancer.target)

#スコア格納
training_accuracy = []
test_accuracy = []

#n_neighborsは1-10まで
n_neighbors_setting = range(1,11)

for n_neighbors in n_neighbors_setting:
    #モデルを構築
    clf = KNeighborsClassifier(n_neighbors = n_neighbors)
    clf.fit(X_train,y_train)
    #訓練データで検証
    training_accuracy.append(clf.score(X_train,y_train))
    #テストデータで検証
    test_accuracy.append(clf.score(X_test,y_test))

plt.plot(n_neighbors_setting,training_accuracy,label = "training")
plt.plot(n_neighbors_setting,test_accuracy,label = "test")
plt.legend()
plt.show()
