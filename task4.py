#------------------------------------------------------------
# This file is to build kNN classifier
#------------------------------------------------------------

#----------------------------------------------  
#     Load Data   
#---------------------------------------------- 
import task1
train_data, train_target, test_data, test_target = task1.getData()

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

START_K, END_K = 2, 7
#----------------------------------------------  
#     Find the best n_neighbors and p   
#----------------------------------------------
from sklearn.model_selection import GridSearchCV

print "Finding the best parameters..."

#----------------------------------------------  
#     Use GridSearchCV to find the best parameters
#----------------------------------------------  

kNN_parameters = [{'n_neighbors': [x for x in range(START_K, END_K)], 'p': [1, 2, 3]}]
   
clf = GridSearchCV(KNeighborsClassifier(), kNN_parameters, cv=10, scoring='accuracy')
clf.fit(train_data, train_target)

print "Best parameters for kNN:"
print clf.best_params_

print "Start building Decision Tree classifier..."
    
print "Score=", clf.score(train_data, train_target)
p = clf.predict(test_data)
print "Evaluation result for kNN:"
print metrics.classification_report(test_target, p)
print metrics.confusion_matrix(test_target, p)

#----------------------------------------------  
#     Draw figure of the distribution of scores  
#----------------------------------------------   
# arr_score = []
# minScore, maxScore, maxIndex = 0, 0, 0
# for i in range (START_K, END_K):
#     print "Processing k=%d..." % i
#     clf = KNeighborsClassifier(n_neighbors=i)
#     clf.fit(test_data, test_target)
#      
#     scores = cross_val_score(clf, test_data, test_target, scoring="accuracy", cv=10)
#     arr_score.append(scores.mean())
#     score_mean = scores.mean()
#     if score_mean > maxScore:
#         maxScore = score_mean
#         maxIndex = i

# import matplotlib.pyplot as plt

# x_depth = [d for d in range(START_K, END_K)]
# y_score = arr_score
#  
# plt.plot(x_depth, y_score, 'ob')
#  
# plt.xlim(START_K, END_K)
# plt.ylim(minScore, maxScore)
#  
# plt.xlabel('K Value')
# plt.ylabel('Score')
# plt.title("K Nearest Neighbors Score")
# plt.show()

print "Done."
