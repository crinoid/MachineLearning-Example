#------------------------------------------------------------
# This file is to build Decision classifier
#------------------------------------------------------------
from sklearn import tree, metrics
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
#----------------------------------------------  
#     Load Data   
#---------------------------------------------- 
import task1
train_data, train_target, test_data, test_target = task1.getData()

START_DEPTH, END_DEPTH = 30, 61
#----------------------------------------------  
#     Find the best depth   
#----------------------------------------------
print "Start finding best depth..."  

arr_score = []
minScore, maxScore, maxIndex = 0, 0, 0

for i in range(START_DEPTH, END_DEPTH):
    print "Processing depth %d..." % i
    
    clf = tree.DecisionTreeClassifier(max_depth=i)
    clf = clf.fit(train_data, train_target)
 
    scores = cross_val_score(clf, train_data, train_target, scoring="accuracy", cv=10, verbose=20, n_jobs=-1)
    score_mean = scores.mean()
    arr_score.append(score_mean)

    if score_mean > maxScore:
        maxScore = score_mean
        maxIndex = i
        
print "Best depth=%d" % maxIndex
print "Score=%f" % maxScore

#----------------------------------------------  
#     Draw figure of the distribution of scores  
#----------------------------------------------   
x_depth = [d for d in range(START_DEPTH, END_DEPTH)]
y_score = arr_score
 
plt.plot(x_depth, y_score, 'ob')  # draw blue dots
 
plt.xlim(START_DEPTH, END_DEPTH)
plt.ylim(minScore, maxScore)
 
plt.xlabel('Depth')
plt.ylabel('Score')
plt.title("Decision Tree Depth Score")
plt.show()

print "Start building Decision Tree classifier..."
#----------------------------------------------  
#     Build DT classifier   
#----------------------------------------------  
clf = tree.DecisionTreeClassifier(max_depth=maxIndex)
clf = clf.fit(train_data, train_target)
p = clf.predict(test_data)
print "Evaluation result for DT:"
print metrics.classification_report(test_target, p)
print metrics.confusion_matrix(test_target, p)

print "Done."
