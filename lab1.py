import numpy as np
import pandas as pd
from time import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

#read the given train dataset
col_names=['gameDuration','winner','firstBlood','firstTower','firstInhibitor','firstBaron','firstDragon','firstRiftHerald','t1_towerKills','t1_inhibitorKills','t1_baronKills','t1_dragonKills','t1_riftHeraldKills','t2_towerKills','t2_inhibitorKills','t2_baronKills','t2_dragonKills','t2_riftHeraldKills']
pima= pd.read_csv("new_data.csv",header=None, names=col_names) 
pima=pima.iloc[1:]
pima.head()
feature_cols=['gameDuration','firstBlood','firstTower','firstInhibitor','firstBaron','firstDragon','firstRiftHerald','t1_towerKills','t1_inhibitorKills','t1_baronKills','t1_dragonKills','t1_riftHeraldKills','t2_towerKills','t2_inhibitorKills','t2_baronKills','t2_dragonKills','t2_riftHeraldKills']
X=pima[feature_cols]
y=pima.winner

#split training and testing set
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.25,random_state=1)

#list for storing accuracy 
score=[]

#decision tree
begin_time = time() #计时模块
clf=DecisionTreeClassifier(criterion="gini",max_depth=18,min_samples_split=15)
clf=clf.fit(X_train,y_train)
y_pred1=clf.predict(X_test)
print("Accuracy of dt:",round(accuracy_score(y_test,y_pred1),4))
score.append(round(accuracy_score(y_test,y_pred1),4))
end_time = time() #计时模块
run_time1 = end_time-begin_time #计时模块
print ('running this program costs：',run_time1,'s') #计时模块

#knn
begin_time = time() #计时模块
knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train, y_train)
knn.score(X_test,y_test)
knn_para=knn.get_params()
a3=round(knn.score(X_test, y_test),4)
print("Accuracy of knn:",a3)
score.append(a3)
end_time = time() #计时模块
run_time2 = end_time-begin_time #计时模块
print ('running this program costs：',run_time2,'s') #计时模块

#mlp
begin_time = time() #计时模块
mlp = MLPClassifier(hidden_layer_sizes=18, max_iter=1000, random_state=None)
mlp.fit(X_train, y_train)
mlp_para=mlp.get_params()
a4=round(mlp.score(X_test, y_test),4)
print("Accuracy of mlp:",a4)
score.append(a4)
end_time = time() #计时模块
run_time3 = end_time-begin_time #计时模块
print ('running this program costs：',run_time3,'s') #计时模块

#fusion method
begin_time = time() #计时模块
pre_dt=clf.predict_proba(X)
pre_knn=knn.predict_proba(X)
pre_mlp=mlp.predict_proba(X)
pre_mean=pre_dt+pre_knn+pre_mlp
mean=[]
for i in range(len(pre_mean)):
    if pre_mean[i][0]<pre_mean[i][1]:
        mean.append('2')
    else :
        mean.append('1')
maxs=[]
for i in range(len(pre_mean)):
    decide=[pre_knn[i][1],pre_dt[i][1],pre_mlp[i][1]]
    max=decide[np.argmax(decide)]
    if max>0.5:
        maxs.append('2')
    else:
        maxs.append('1')
    
max_ii=0
mean_ii=0
for k in range(len(pre_mean)):
    if  (mean[k]==y[k]):
        mean_ii=mean_ii+1
    if (maxs[k]==y[k]):
        max_ii=max_ii+1

amax=max_ii/len(pre_mean)
amax=round(amax,4)
print("Accuracy of fusion method of maximum value:",amax)
score.append(amax)
amean=mean_ii/len(pre_mean)
amean=round(amean,4)
score.append(amean) 
end_time = time() #计时模块
run_time = end_time-begin_time #计时模块
print ('running these two programs costs：',run_time1+run_time2+run_time3+run_time/2,'s') 
print("Accuracy of fusion method of mean value:",amean)
print ('running these two programs costs：',run_time1+run_time2+run_time3+run_time/2,'s') #计时模块

#predict for the test dateset
pi= pd.read_csv("test_set.csv",header=None, names=col_names) 
pi=pi.iloc[1:]
pima.head()
X=pi[feature_cols]
testresult=clf.predict(X)
print(testresult)


