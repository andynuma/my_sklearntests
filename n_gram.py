from sklearn.feature_extraction.text import CountVectorizer

bards_words = ["The fool doth think he is wise","but the wise man knows himself to be a fool"]

#n-gram
n = 2

#ngram_rangeで連続するトークンの数を指定、一つ目の値は最小長、二つ目の値は最大長
cv = CountVectorizer(ngram_range=(1,n)).fit(bards_words)
#cv.vocabulary_で割り当てをみる
print("---------cv.vocabulary_-----------:",cv.vocabulary_)
#cv/get_feature_namesで語彙をみる
print("----------cv.feature_names---------:",cv.get_feature_names())
#transformして出現頻度を見てみる
#例えば一つ目の配列には"The fool doth think he is wise"について、
#出現しているfeature_namesのインデックスが1になっている
print(cv.transform(bards_words).toarray())
