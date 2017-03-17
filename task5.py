#------------------------------------------------------------
# This file is to build SVM classifier
#------------------------------------------------------------
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn import svm
#----------------------------------------------  
#     Load Data   
#----------------------------------------------  
import task1
train_data, train_target, test_data, test_target = task1.getData()

MAX_SIZE = 10000
START_C, END_C = -4, 5

print "Finding the best parameters..."

#----------------------------------------------  
#     Use GridSearchCV to find the best parameters
#----------------------------------------------  
 
svm_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [10 ** i for i in range(START_C, END_C)]},
                    {'kernel': ['linear'], 'C': [10 ** i for i in range(START_C, END_C)]}]
   
clf = GridSearchCV(svm.SVC(), svm_parameters, cv=10, scoring='accuracy')
clf.fit(train_data[:MAX_SIZE], train_target[:MAX_SIZE])

print "Best parameters for SVM:"
print clf.best_params_

print "Start building SVM classifier..."

print "Score=", clf.score(train_data[:MAX_SIZE], train_target[:MAX_SIZE])
p = clf.predict(test_data) 
print "Evaluation result for SVM" 
print metrics.classification_report(test_target, p)
print metrics.confusion_matrix(test_target, p)

#----------------------------------------------  
#     Draw figure of the distribution of scores  
#---------------------------------------------- 
# arr_score=[]
# maxValue,maxIndex=0,0;
# 
# for i in range(-START_C, END_C):
#     clf = svm.SVC(C=10**i,kernel="linear")  
#     clf.fit(train_data[:MAX_SIZE],train_target[:MAX_SIZE])
#     scores_train = cross_val_score(clf, train_data[:MAX_SIZE], train_target[:MAX_SIZE],scoring="accuracy", cv=5, verbose=20, n_jobs=-1)
#     score_mean= scores_train.mean()
#     arr_score.append(score_mean)
#     if score_mean>maxValue:
#         maxValue=score_mean
#         maxIndex=i
#         
# import matplotlib.pyplot as plt        
# x_depth = [d for d in range(START_C, END_C)]
# y_score = arr_score       
# plt.plot(x_depth, y_score,'ob')
# 
# plt.xlim(START_C-1, END_C)
# plt.ylim(0,maxValue)
# 
# plt.xlabel('C value (10^x)')
# plt.ylabel('Score')
# plt.title("SVM Score")
# plt.show()

print "Done."


