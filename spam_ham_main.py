"""
coding: utf-8
INFX574 Anna Huang Tiffany Chiu
PS4
"""

# import necessary libraries
import mailbox
import re
import collections
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import sklearn
from textblob import TextBlob
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
#nltk.download('wordnet') #for lemma
#nltk.download('stopwords') #for stopwords removal
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer() 
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split 
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.metrics import roc_curve, auc


# read .mbox files into python using mailbox library
hambox = mailbox.mbox('Ham.mbox')
spambox = mailbox.mbox('Spam.mbox')
spambox1 = mailbox.mbox('Spam1.mbox')
spambox2 = mailbox.mbox('Spam2.mbox')   
spambox3 = mailbox.mbox('Spam3.mbox')
spambox4 = mailbox.mbox('Spam4.mbox')
spambox5 = mailbox.mbox('Spam5.mbox')

# see how many messages each file has
print("hambox has", len(hambox), "messages")
print("spambox has", len(spambox), "messages")
print("spambox1 has", len(spambox1), "messages")
print("spambox2 has", len(spambox2), "messages")
print("spambox3 has", len(spambox3), "messages")
print("spambox4 has", len(spambox4), "messages")
print("spambox5 has", len(spambox5), "messages")

# see total messages for ham and spam before pre-processing
print("ham has total of", len(hambox), "messages")

spamtotal = 192+172+84+27+76+415
print("spam has total of", spamtotal, "messages")

# create function to parse .mbox format to dataframe format
def parse_mbox_to_df (mbox, label):
    
    """
    function parse .mbox items
    return a dataframe of words by email with labels
    ------
    input: mbox, label
    type: .mbox, str
    output: pandas dataframe
    ------
    example:
    hambox = mailbox.mbox('data/Ham.mbox')
    df_ham = parse_mbox_to_df (hambox, label='ham')
    """
    
    # create empty content list to append plain text email messages
    content = []
    
    # get plain text 'email body'
    for message in mbox:
        # use is.multipart to take care of sub-messages
        if message.is_multipart():
            # use walk to iterate and return parts and subparts of message
            for part in message.walk():
                if part.is_multipart():
                    for subpart in part.walk():
                        # check message content type for text/plain
                        if subpart.get_content_type() == 'text/plain':
                            body = str(subpart.get_payload(decode=True))
                            body = re.sub(r'http\S+', '', body)
                            # append messages to content list
                            content.append(body.strip().replace('\\n', ' ').replace('\r', ''))
                elif part.get_content_type() == 'text/plain':
                    body = str(part.get_payload(decode=True))
                    body = re.sub(r'http\S+', '', body)
                    content.append(body.strip().replace('\\n', ' ').replace('\r', ''))
                    
        elif message.get_content_type() == 'text/plain':
            body = str(message.get_payload(decode=True))
            body = re.sub(r'http\S+', '',body)
            content.append(body.strip().replace('\\n', ' ').replace('\r', ''))
        
    for j in range(len(content)):
        content[j] = [i for i in re.compile('\w+').findall(content[j]) if len(i)>1]
    
    dict = {'content': content, 'label':label}
    df = pd.DataFrame(dict)
    return df

# convert to dataframes and create labels for the ham and spam dataframes
df_ham = parse_mbox_to_df (hambox, label='ham')
df_spam = parse_mbox_to_df (spambox, label='spam')
df_spam1 = parse_mbox_to_df (spambox1, label='spam')
df_spam2 = parse_mbox_to_df (spambox2, label='spam')
df_spam3 = parse_mbox_to_df (spambox3, label='spam')
df_spam4 = parse_mbox_to_df (spambox4, label='spam')
df_spam5 = parse_mbox_to_df (spambox5, label='spam')

# combine ham and spam dataframes into one dataframe
df = pd.concat([df_ham,df_spam,df_spam1,df_spam2,
                df_spam3, df_spam4, df_spam5]
                ,ignore_index=True)


# get rid of stopwords and perform lemmatization
# create empty words list to append pre-processed messages 
words = []
for item in df['content']:
    # change all words to lowercase
    item_1 = [word.lower() for word in item]
    # remove all English stopwords
    item_2 = [word for word in item_1 if word not in stopwords.words('english')]
    # lemmatize words
    item_3 = [lmtzr.lemmatize(word)for word in item_2]
    words.append(tuple(item_3))

# remove duplicate messages
df['words'] = words
df = df.drop(['content'],axis = 1)
df = df.drop_duplicates().reset_index()

# delete unnecessary columns
del df['index']

#df[(df.label == 'ham')]
# 773 hams after cleaning
#df[(df.label == 'spam')]
# 483 spams after cleaning

# change words column from tuple to list
for i in range(len(df['words'])):
    df['words'][i] = ' '.join(list(df['words'][i]))   

# tokenize and build vocabulary
vectorizer = CountVectorizer()
vectorizer.fit(df['words'])

# summarize
#print(vectorizer.vocabulary_).head

# encode document
vector = vectorizer.transform(df['words'])
#print(vector)

# summarize encoded vector
print("Sparse matrix shape:",vector.shape)
print("Data type:",type(vector))
print("Number of non-zeros:", vector.nnz)
print("Sparsity: %.2f%%" % (100.0 * vector.nnz /(vector.shape[0] * vector.shape[1])))
print("Feature vector:",vector.toarray())
"""
result:
Sparse matrix shape: (1256, 17080)
Data type: <class 'scipy.sparse.csr.csr_matrix'>
Number of non-zeros: 143264
Sparsity: 0.67%
"""

# weight terms and normalize
tfidf_transformer = TfidfTransformer().fit(vector)
messages_tfidf = tfidf_transformer.transform(vector)
print(messages_tfidf.shape)
"""
(1256, 17080)
"""

# split data into train and test
msg_train, msg_test, label_train, label_test = train_test_split(df['words'], df['label'], test_size=0.2)

print("training set length:", len(msg_train))
print("test set length:", len(msg_test))
print("total length:", len(msg_train) + len(msg_test))
"""
training set length: 1004
test set length: 252
total length: 1256
"""

# Spam Filtering Algorithm #1: Below is the code and results on Naive Bayes algorithm.

# put previous steps into pipeline, which includes transforms and estimator
# include frequency count, weighted TF-IDF scores, NB classifier
# pipeline allows assembling many steps to be cross-validated together, with parameters
pipeline_NB = Pipeline([
    ('count', CountVectorizer()), 
    ('tfidf', TfidfTransformer()), 
    ('classifier', MultinomialNB()), 
])


# calculate cross validation scores using cross_val_score
# split data randomly into 10 parts: 9 for training, 1 for scoring
# use accuracy scoring metric
# use -1 for n_jobs to indivate using all cores, for faster processing
scores = cross_val_score(pipeline_NB,  
                         msg_train,  
                         label_train,  
                         cv = 10,  
                         scoring = 'accuracy',  
                         n_jobs = -1, 
                         )

print("mean scores:", scores.mean())
print("standard deviation scores:", scores.std())
print("scores:", scores)
"""
result:
mean scores: 0.793700058241
standard deviation scores: 0.0340068435106
"""

# The accuracy scores seem to center around 0.78, 
#which is not great but also not terrible. 
#There is a standard deviation of around +/- 0.04.

# tune parameters for cross validation
params = {
    'tfidf__use_idf': (True, False),
}

# use GridSearchCV to search over specified param for an estimator
# enable refit to use available data at the end on the best found parameter combination
# n_jobs = -1 to use all cores in processing
# use accuracy scoring metric
# use stratified k-fold cross validation
grid = GridSearchCV(
    pipeline_NB,
    params,
    refit = True,  
    n_jobs = -1,  
    scoring = 'accuracy',  
    cv = StratifiedKFold(5,True),
)

# fit naive bayes cross validation with new parameters
get_ipython().run_line_magic('time', 'nb_detector = grid.fit(msg_train, label_train)')
predictions = nb_detector.predict(msg_test)
# see predictions result for test set
print(confusion_matrix(label_test, predictions))
print(classification_report(label_test, predictions))
"""
result:
Wall time: 4.37 s
[[151   9]
 [ 53  39]]
             precision    recall  f1-score   support

        ham       0.74      0.94      0.83       160
       spam       0.81      0.42      0.56        92

avg / total       0.77      0.75      0.73       252
"""

# Spam Filtering Algorithm #2: Below are codes and results for SVM (support vector machines).

# put previous steps into pipeline, which includes transforms and estimator
# include frequency count, weighted TF-IDF scores, support vector classification
# pipeline allows assembling many steps to be cross-validated together, with parameters
pipeline_svm = Pipeline([
    ('bow', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('classifier', SVC()),
])

# set parameters
param_svm = [
  {'classifier__C': [1, 10, 100, 1000], 'classifier__kernel': ['linear']},
  {'classifier__C': [1, 10, 100, 1000], 
   'classifier__gamma': [0.001, 0.0001], 'classifier__kernel': ['rbf']},
]

# use GridSearchCV to search over specified param for an estimator
# enable refit to use available data at the end on the best detected classifier
# n_jobs = -1 to use all cores in processing
# use accuracy scoring metric
# use stratified k-fold cross validation
grid_svm = GridSearchCV(
    pipeline_svm,  
    param_grid=param_svm,  
    refit=True, 
    n_jobs=-1,  
    scoring='accuracy', 
    cv=StratifiedKFold(5,True),  
)

# fit SVM cross validation with new parameters to find best combination from param_svm
svm_detector = grid_svm.fit(msg_train, label_train) # find the best combination from param_svm')

# show main classification metrics in a text report
print(confusion_matrix(label_test, svm_detector.predict(msg_test)))
print (classification_report(label_test, svm_detector.predict(msg_test)))
"""
result:
Wall time: 40.3 s
[[134  26]
 [  7  85]]
             precision    recall  f1-score   support

        ham       0.95      0.84      0.89       160
       spam       0.77      0.92      0.84        92

avg / total       0.88      0.87      0.87       252
"""

# calculate cross validation scores using cross_val_score
# split data randomly into 10 parts: 9 for training, 1 for scoring
# use accuracy scoring metric
# use -1 for n_jobs to indivate using all cores, for faster processing
scores = cross_val_score(pipeline_svm,  
                         msg_train,  
                         label_train,  
                         cv = 10,  
                         scoring = 'accuracy',  
                         n_jobs = -1, 
                         )

print("mean scores:", scores.mean())
print("standard deviation scores:", scores.std())
print("scores:", scores)


"""
The accuracy scores here center around 0.62, 
which is much lower than that of Naive Bayes. 
However, the standard deviation is a lot smaller here, at +/- 0.0011.
"""

# Spam Filtering Algorithm #3: Below are codes and results for logistic regression.

# use sklearn's LogisticRegression function
# use msg_train and label_train as variables
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(msg_train)
classifier = LogisticRegression()
classifier.fit(X_train, label_train)

# see predictions result for test set
X_test = vectorizer.transform(msg_test)
predictions = classifier.predict(X_test)
print(confusion_matrix(label_test, predictions))
print(classification_report(label_test, predictions))
"""
result:
[[142  18]
 [ 20  72]]
             precision    recall  f1-score   support

        ham       0.88      0.89      0.88       160
       spam       0.80      0.78      0.79        92

avg / total       0.85      0.85      0.85       252
""""

# calculate accuracy
print("Accuracy:", metrics.accuracy_score(label_test, predictions))
"""
Accuracy: 0.849206349206
"""
