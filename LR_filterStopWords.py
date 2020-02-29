import glob
import sys
import numpy as np


path = str(sys.argv[1])
trainspamfilepath=path+'/train/spam/*.txt'
trainhamfilepath=path+'/train/ham/*.txt'
testspamfilepath=path+'/test/spam/*.txt'
testhamfilepath=path+'/test/ham/*.txt'
stopwordspath=str(sys.argv[2])


iterations = 100
regularization_parameter = 0.1
learning_rate = 0.01

#------get all stop words----------------
stopWords=[]
with open(stopwordspath, 'r') as f:
    stopWords = f.read().split()

#-----------------Read data of spam and ham------------------------

def getData(path1, path2):
    labels = []
    files = glob.glob(path1)
    word_dict={}
    no_of_files=0

    
    for file in files:
        no_of_files+=1
        labels.extend([1])
        with open(file, 'r', encoding="utf8", errors="ignore") as contents:
            content = contents.read().split()
            for word in content:
                if word not in stopWords:#Exclude stopwords
                    word_dict[word]=1
                    
    files = glob.glob(path2)
    for file in files:
        no_of_files+=1
        labels.extend([0])
        with open(file, 'r', encoding="utf8", errors="ignore") as contents:
            content = contents.read().split()
            for word in content:
                if word not in stopWords:
                    word_dict[word]=1
                
    return word_dict, no_of_files, labels

#--Create the features matrix: rows=files, cols=unique words, entries=count(word)---

def fill_matrix(fmat, uwords, path1, path2):
    files = glob.glob(path1)
    i=0
    for file in files:
        with open(file, 'r', encoding="utf8", errors="ignore") as contents:
            content = contents.read().split()
            for word in content:
                if word not in stopWords:
                    fmat[i][uwords.index(word)] = content.count(word)
        i+=1
        
    files = glob.glob(path2)
    for file in files:
        with open(file, 'r', encoding="utf8", errors="ignore") as contents:
            content = contents.read().split()
            for word in content:
                if word not in stopWords:
                    fmat[i][uwords.index(word)] = content.count(word)
        i+=1
                
#--Create the features matrix for test: cols=unique words in train set--

def fill_matrix_test(fmat, uwords, path1, path2):
    files = glob.glob(path1)
    i=0
    for file in files:
        with open(file, 'r', encoding="utf8", errors="ignore") as contents:
            content = contents.read().split()
            for word in content:
                if word not in stopWords:
                    if word in uwords:
                        fmat[i][uwords.index(word)] = content.count(word)
        i+=1
        
    files = glob.glob(path2)
    for file in files:
        with open(file, 'r', encoding="utf8", errors="ignore") as contents:
            content = contents.read().split()
            for word in content:
                if word not in stopWords:
                    if word in uwords:
                        fmat[i][uwords.index(word)] = content.count(word)
        i+=1
        
        
        
        
def sigmoid(z):
    return 1.0/(1+np.exp(-z))


def calc_sigmoid(sigmoids, f_mat, theta):
    sigmoids=sigmoid(f_mat * theta)
    return sigmoids
        
def update_theta(theta):
    diff = f_mat.transpose()*(tr_labels-sigmoids)
    theta = theta +(learning_rate * (diff - (regularization_parameter * theta)))
    return theta

#----------------------Training---------------------------------------------

train_uniq_words, train_file_count, train_labels = getData(trainspamfilepath, trainhamfilepath)
f_mat = np.zeros((train_file_count, len(train_uniq_words)))
theta = np.matrix(np.zeros((len(train_uniq_words),1),dtype=float))
sigmoids = np.matrix(np.zeros((train_file_count,1),dtype=float))
train_uniq_list = list(train_uniq_words.keys())
fill_matrix(f_mat, train_uniq_list, trainspamfilepath, trainhamfilepath)

tr_labels = np.array(train_labels).reshape((train_file_count,1))


for i in range(iterations):
    sigmoids=calc_sigmoid(sigmoids, f_mat, theta)
    theta=update_theta(theta)

#----------------------Testing--------------------------------------------
test_uniq_words, test_file_count, test_labels = getData(testspamfilepath, testhamfilepath)
test_f_mat = np.zeros((test_file_count, len(train_uniq_words)))
fill_matrix_test(test_f_mat, train_uniq_list, testspamfilepath, testhamfilepath)

right_spam=wrong_spam=right_ham=wrong_ham=0

result = test_f_mat * theta

for f in range(test_file_count):
    if sigmoid(result[f][0])>0.5:
        if test_labels[f]==1:
            right_spam+=1
        else:
            wrong_spam+=1
    else:
        if test_labels[f]==0:
            right_ham+=1
        else:
            wrong_ham+=1
            
print("Spam Accuracy with stop words elimination:",(right_spam/(right_spam+wrong_spam) * 100))
print("Ham Accuracy with stop words elimination:",(right_ham/(right_ham+wrong_ham) * 100))
print("Net Accuracy:",((right_spam+right_ham)/test_file_count)*100)


            
            
            



