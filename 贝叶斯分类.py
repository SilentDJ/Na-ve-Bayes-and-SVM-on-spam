# coding=UTF-8
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
def preprocessing(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)] #nltk.word_tokenize进行分词
    stops = stopwords.words('english')                 #停止词，是由英文单词:stopword翻译过来的，原来在英语里面会遇到很多a，the，or等使用频率很多的字或词，
                                                        # 常为冠词、介词、副词或连词等。如果搜索引擎要将这些词都索引的话，那么几乎每个网站都会被索引，也就是说工作量巨大
    tokens = [token for token in tokens if token not in stops]

    tokens = [token.lower() for token in tokens if len(token) >= 3]
    lmtzr = WordNetLemmatizer()                                       #词性还原
    tokens = [lmtzr.lemmatize(token) for token in tokens]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text
import csv
file_path=r'F:\smsspamcollection\list_son.txt'
sms = open(file_path,'r',encoding='utf-8')
sms_data = []
sms_label = []
csv_reader = csv.reader(sms,delimiter='\t')
# 将数据分别存入数据列表和目标分类列表
for line in csv_reader:
    sms_label.append(line[0])
    sms_data.append(preprocessing(line[1]))
sms.close()
print(sms_data )
print(sms_label)

# 将数据分为训练集和测试集
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = \
    train_test_split(sms_data,sms_label,
                     test_size=0.3,random_state=0, shuffle=True)

# 建立数据的特征向量
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
X_train = tfidf.fit_transform(x_train)
X_test = tfidf.transform(x_test)


# 建立NB模型
from sklearn.naive_bayes import MultinomialNB
module = MultinomialNB().fit(X_train, y_train)
y_nb_pred = module.predict(X_test)


# 输出模型分类的各个指标
from sklearn.metrics import classification_report
cr = classification_report(y_nb_pred,y_test)
print(cr)

# 建立SVC模型
from sklearn.svm import LinearSVC
module = LinearSVC().fit(X_train, y_train)
y_svm_pred = module.predict(X_test)
#ceshimoxing
from sklearn.metrics import confusion_matrix
print('svm_confusion_matrix:')
cm = confusion_matrix(y_test, y_svm_pred)
print(cm)
print('svm_classification_report:')
print(classification_report(y_test, y_svm_pred))