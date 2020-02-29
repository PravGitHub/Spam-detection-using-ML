import sys
import glob
import math

path = str(sys.argv[1])
trainspamfilepath=path+'/train/spam/*.txt'
trainhamfilepath=path+'/train/ham/*.txt'
testspamfilepath=path+'/test/spam/*.txt'
testhamfilepath=path+'/test/ham/*.txt'

#------------read data, generate word dictionary---------

def getData(folder):
    files = glob.glob(folder)
    word_dict={}
    no_of_files=0;
    for file in files:
        no_of_files+=1
        with open(file, 'r', encoding="utf8", errors="ignore") as contents:
            content = contents.read().split()
            for word in content:
                if word in word_dict.keys():
                    word_dict[word]+=1
                else:
                    word_dict[word]=1
                    
    return word_dict, no_of_files

def countData(dictionary):
    tot_count=0
    for key in dictionary.keys():
        tot_count+=dictionary[key]
    
    return tot_count
        


def condProbability(dictionary, count, uniq):
    prob_dict = {}
    for word in dictionary.keys():
        word_count = dictionary[word]+1
        prob_dict[word] = word_count/(count+uniq)
    return prob_dict
    
#--------classify spam and ham in testing data---------------

def classify(folder, spam_dict, ham_dict, dinom_spam, dinom_ham, prior):
    files = glob.glob(folder)
    result = {'ham':0 , 'spam':0}
    for file in files:
        p_spam = math.log(prior['spam'])
        p_ham = math.log(prior['ham'])
        with open(file, 'r',encoding="utf8",errors="ignore") as contents:
            content = contents.read().split()
            for word in content:
                if word in spam_dict.keys():
                    p_spam+=math.log(spam_dict[word])
                else:
                    p_spam+=math.log(1/dinom_spam)
                    
                if word in ham_dict.keys():
                    p_ham+=math.log(ham_dict[word])
                else:
                    p_ham+=math.log(1/dinom_ham)
                
        if p_spam >= p_ham:
            result['spam']+=1
        else:
            result['ham']+=1
    
    return result
                
train_spam_dict, no_spam_files = getData(trainspamfilepath)
train_ham_dict, no_ham_files = getData(trainhamfilepath)

prior={}
prior['ham']=(no_ham_files)/(no_ham_files+no_spam_files)
prior['spam']=(no_spam_files)/(no_ham_files+no_spam_files)

spam_word_count = countData(train_spam_dict)
ham_word_count = countData(train_ham_dict)

uniq_count = len(set(list(train_spam_dict.keys()) + list(train_ham_dict.keys())))

spam_prob_dict = condProbability(train_spam_dict, spam_word_count, uniq_count)
ham_prob_dict = condProbability(train_ham_dict, ham_word_count, uniq_count)

dinom_spam = spam_word_count + uniq_count
dinom_ham = ham_word_count + uniq_count
output_spam_dict = classify(testspamfilepath, spam_prob_dict, ham_prob_dict, dinom_spam, dinom_ham, prior)
output_ham_dict = classify(testhamfilepath, spam_prob_dict, ham_prob_dict, dinom_spam, dinom_ham, prior)

testspamlen=testhamlen=0
files = glob.glob(testspamfilepath)
for file in files:
    testspamlen+=1

files = glob.glob(testhamfilepath)
for file in files:
    testhamlen+=1

print("spam accuracy without stopwords elimination:",(output_spam_dict['spam']/testspamlen)*100)
print("Ham accuracy without stopwords elimination:",(output_ham_dict['ham']/testhamlen)*100)
print("Net Accuracy:",((output_spam_dict['spam']+output_ham_dict['ham'])/(testspamlen+testhamlen) * 100))


