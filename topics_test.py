from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

#トレーニングデータセット
text_train = ["The fool is me","you are sun of the bitch"]

#改行文字削除
#text_train = [doc.replace(b"<br />",b" ") for doc in text_train]


#モデル構築
vect = CountVectorizer(max_features=10000)
X = vect.fit_transform(text_train)

#LDAでトピック分類
from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_topics=10,learning_method="batch",max_iter=25,random_state=0)
document_topics = lda.fit_transform(X)

#それぞれのトピックに対して、特徴量を昇順でソート
#ソートを降順にするために[:,::-1]で行を反転
sorting = np.argsort(lda.components_,axis=1)[:,::-1]
#vectorizerから特徴量名を取得
feature_names = np.array(vect.get_feature_names())

import mglearn

#可視化
mglearn.tools.print_topics(topics=range(10),feature_names=feature_names,sorting=sorting,n_words=10)
