#------------------------------------------------------------
# This file is to build Naive Bayes classifier
#------------------------------------------------------------
from sklearn import metrics
#----------------------------------------------  
#     Load Data   
#---------------------------------------------- 
import task1
train_data, train_target, test_data, test_target = task1.getData()  

print "Start building GaussianNB classifier..."

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
p = clf.fit(train_data, train_target).predict(test_data) 
print "Evaluation result for GaussianNB:"
print metrics.classification_report(test_target, p)
print metrics.confusion_matrix(test_target, p)

print "Build BernoulliNB classifier..."
   
from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB()
p = clf.fit(train_data, train_target).predict(test_data)    
print "Evaluation result for BernoulliNB:"
print metrics.classification_report(test_target, p)
print metrics.confusion_matrix(test_target, p)

print "Done."
