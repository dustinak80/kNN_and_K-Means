# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 07:43:28 2019

@author: Raul Paz/Dustin/Stephen
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NeighborhoodComponentsAnalysis as NCA
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import time
import copy

times = [time.time()]
fsz=(12,8)

def get_data(file):
    """Create a function for loading the csv and editing the file the way we want
    for this excel format
    """
    
    import pandas as pd
    
    #Import csv as a Data Frame, drop columns do now want, drop rows with value na
    data = pd.read_csv(file)
    data = data.drop(['fontVariant', 'm_label',  'orientation', 'm_top', 'm_left', 'originalH', 'originalW', 'h', 'w' ], axis = 1)
    data.dropna()
    print(data.shape)
    
    data = data.loc[(data.strength ==.4) & (data.italic == 0)]
       
    return data

#load the files and assign to class
file = r'C:\Users\Dustin\Desktop\Masters Program\Fall Semester\aaaaStatistical Learning and Data mining\Homework and Readings\Homework\HW2\CALIBRI.csv'
cl2 = get_data(file)
cl2['font'] = 2

file = r'C:\Users\Dustin\Desktop\Masters Program\Fall Semester\aaaaStatistical Learning and Data mining\Homework and Readings\Homework\HW2\COURIER.csv'
cl1 = get_data(file)
cl1['font'] = 1

file = r'C:\Users\Dustin\Desktop\Masters Program\Fall Semester\aaaaStatistical Learning and Data mining\Homework and Readings\Homework\HW2\TIMES.csv'
cl3 = get_data(file)
cl3['font'] = 3

fonts = {1:'COURIER', 2:'CALIBRI', 3:'TIMES'}

del file

#Print the sizes
n_cl1 = cl1.shape; print( n_cl1)
n_cl2 = cl2.shape; print( n_cl2)
n_cl3 = cl3.shape; print( n_cl3)
v = 400

#combine the three classes for full set of Data
data=pd.concat([cl1,cl2,cl3],axis=0)
data.index = range(len(data))
n_data = data.shape; print(n_data)
if n_data[0] == (n_cl1[0]+n_cl2[0]+n_cl3[0]): #check
    print('\nLooks good')

## mean and standard deviation
m=data[data.columns[3:]].mean() #Mean
sd=data[data.columns[3:]].std() #Standard Deviation

## Centralize and stadardize the matrix
data_y = data[data.columns[0]] #Get your y's
data_s = (data[data.columns[3:]]-m)/sd #Centralizing and Standardizing the Data
#print(data_s.mean(), data_s.std())   #Check the values , m=0 sd=1
""" just for visualization of centralized data
#plot of centralized and standardized data
fig1, ax1 = plt.subplots(figsize=fsz)
ax1.set_title('Samples of Shaped Data')
ax1.boxplot([data_s[data_s.columns[3]],data_s[data_s.columns[4]],data_s[data_s.columns[5]]], meanline = True, showmeans=True, labels = ['r0c0','r0c1','r0c2'])
"""

## correlation matrix
corr_m = data_s.corr()
eigs = np.linalg.eig(corr_m)

## eigen values
eig_value = eigs[0]
eig_vecto = eigs[1] 
print(sum(eig_vecto[:,0]**2))
print('\n Sum of eig: ', sum(eig_value))

## Plots
#plot eig_values
fig2, ax2 = plt.subplots(figsize=fsz)
ax2.set_title('Eigenvalues')
ax2.set_ylabel('Eignevalues')
ax2.scatter(range(len(eig_value)), eig_value, alpha=0.5)

"""
Question 1
"""
print('\nQuestion 1\n')

times += [time.time()]

## Rj
ratio = np.cumsum(eig_value)/sum(eig_value)
# Where does Rj>.35 and Rj>60%
a_idx = np.where(ratio>.35)[0][0] #gets the index
b_idx = np.where(ratio>.60)[0][0] #gets the index
a = ratio[a_idx] #gets the value at index
b = ratio[b_idx] #gets the value at index
print('At eigenvalue', (a_idx+1), format(eig_value[a_idx], '.2f'), 'we get a ratio of', format(a, '.2%'),
      'and at', (b_idx+1), format(eig_value[b_idx], '.2f'), 'we get a ratio of', format(b, '.2%'))

#Plot Rj
fig3, ax3 = plt.subplots(figsize=fsz)
ax3.set_title('Ratio Trend')
ax3.set_ylabel('Ratio')
ax3.scatter(range(len(ratio)), ratio, c='b', alpha=.2)
ax3.scatter(a_idx, a, c='green', alpha=1)
ax3.scatter(b_idx, b, c='green', alpha=1)
ax3.plot(range(len(ratio)), [.35]*400, 'r--', alpha=.2)
ax3.plot(range(len(ratio)), [.60]*400, 'r--', alpha=.2)
ax3.text(350,a,'35% Line', horizontalalignment = 'right', verticalalignment = 'bottom')
ax3.text(350,b,'60% Line', horizontalalignment = 'right', verticalalignment = 'bottom')
ax3.text(a_idx,.1,'Eigenvalue 4 > 35% and Eigenvalue 16 > 60%', verticalalignment = 'bottom')

#split the data into test/train
def train_test_split(x,y,train, classification):
    """
    Inputs: All of your features as x, all of your results as y, your train %,
            a list of your classifications
    Outputs:train_data, test_data
    This function will take your data set and shuffle it, then split it by 
    classification evenly to make up your train and test data set    
    """  
    import time
    start = time.time()
    from sklearn.utils import shuffle
    
    y = pd.DataFrame(y)
    x = pd.DataFrame(x)
    xy = pd.concat([y,x], axis=1)
    
    #seperate the groups (class 1,2,...)
    cl = list()
    for i in range(len(classification)):
        group = xy['font'].where(xy['font'] == classification[i]) 
        cl += [group.dropna()]
    #shuffle the groups (seperately)
    for i in range(len(cl)):
        cl[i] = shuffle(cl[i])
        cl[i] = cl[i].reset_index()
    #split each group into test/train
    test_cl = []
    train_cl = []
    check = []
    for i in range(len(cl)):
        num = int(train*len(cl[i])) #needed to make into integer
        train_cl += [cl[i][0:num]]
        test_cl += [cl[i][num:]]
        check += [round(len(train_cl[i])/len(cl[i]),3)]
        print('train', classification[i], 'has a size of', 
              format(check[i], '.2%'), 'compared to data.')
    
    #combine back to complete data frames
    for i in range(len(classification)):
        if i == 0: #start the combining of the data frame with class 1
            train_data = xy.iloc[train_cl[i]['index']]
            test_data = xy.iloc[test_cl[i]['index']]
        else: #keep adding to the dataframe with class 2,3,....
            train_data = pd.concat([train_data,xy.iloc[train_cl[i]['index']]], axis = 0)
            test_data = pd.concat([test_data,xy.iloc[test_cl[i]['index']]], axis = 0)
    end = time.time()
    #reset index
        
    print('The total ratio for train data is', len(train_data)/len(xy), 'of all data')
    print('This function took', end-start)
    return train_data, test_data, check

train = .80 #ratio you want to split
train_data, test_data, check = train_test_split(data_s,data_y,train, list(fonts.keys()))

fig4, ax4 = plt.subplots(figsize=fsz)
ax4.set_title('% of Data is Train')
ax4.set_ylabel('%')
ax4.set_xlabel('class')
ax4.scatter(list(fonts.keys()), check, alpha=0.5, color = 'green')
ax4.plot(list(fonts.keys()), check, alpha=0.5, color = 'green')
ax4.scatter(list(fonts.keys()), [round((1-check[i]),2) for i in range(len(check))], alpha=0.5, color = 'red')
ax4.plot(list(fonts.keys()), [round((1-check[i]),2) for i in range(len(check))], alpha=0.5, color = 'red')

"""
Question 2
"""
print('\nQuestion 2\n')

times += [time.time()]

#project your data

def project(train, test, eig_vector, index):
    """
    This projects whatever is input into the definition onto the eigenvectors
    specified. Make sure indexes are correct!
    """
    #reset Index for merging with project values
    train = train.reset_index(drop = True)  
    test = test.reset_index(drop = True) 
    #project the data
    trainx = pd.DataFrame(np.dot(train[train.columns[1:]],eig_vector))
    testx = pd.DataFrame(np.dot(test[test.columns[1:]],eig_vector))
    #concat the data
    trainy = train[train.columns[0]]
    testy = test[test.columns[0]]
    #get up to that index
    trainx = trainx[trainx.columns[index[0]:index[1]]]
    testx= testx[testx.columns[index[0]:index[1]]]
    
    return trainx, testx, trainy, testy

index = (0,a_idx+1) #index of Rj > 30%
X_train1, X_test1, y_train1, y_test1 = project(train_data,test_data, eig_vecto, index)

nbs1=[5] #neighbors

def KNN(X_train,y_train,X_test, nbs):
    """
    
    """
    pred=[]
    for i in range(len(nbs)):
        neigh = KNC(n_neighbors=nbs[i]) #Built in function that chooses the best method
        neigh.fit(X_train, y_train)
        preda=list(neigh.predict(X_test))
        pred += preda #code with all of the predictions
    
    #print(pred)
    return pred

pred1=KNN(X_train1, y_train1, X_test1, nbs1) #call the funciton

## get to where this works for any values
def percentage(pred, classes, y_test):
    cat = list(classes.keys())
    totals = []
    for i in range(len(cat)):
        l = sum(y_test == cat[i])
        totals += [l]
    #Check that totals of each group sum to length of total
    print('Lengths are:', sum(totals)==len(y_test))

    correct = 0
    wrong = 0
    totals_wrong=[0] * len(totals) #have to fill it with zeros because just adding to index 
    for j in range(len(pred)):
        if (pred[j] == y_test[j]) == True:
            correct += 1
        else:
            wrong += 1
            check = (pred[j] == cat) #which category is pred[j] in
            for i in range(len(check)): #finding the index of the correct value
                if check[i] == 1: #True =1, False = 0
                    totals_wrong[i] += 1
    #check that counts are correct
    print('Count for wrong + correct is:', len(y_test) == correct+wrong)
    tot_perc = correct/len(y_test)
    perc_correct = []
    perc_wrong = []
    for i in range(len(totals)):
        perc_wrong += [round((totals_wrong[i]/totals[i])*100,2)]
        perc_correct += [round((1-(totals_wrong[i]/totals[i]))*100,2)]
    return perc_correct, perc_wrong, tot_perc

#total_per, cal_per, cou_per, tnr_per = percentage(pred, fonts, y_test)
perc_correct, perc_wrong, tot_perc = percentage(pred1, fonts, y_test1)
print('Dimension is:', (index[1]-index[0])) #dimensions of data
print('Percent Correct: \n{0} \nMean Percent Correct: {1:.2f}'.format(perc_correct, tot_perc))

#Confusion Matrix
def confusion_table(y_test, predictions):
    """
    """
    conf_m = confusion_matrix(y_test, predictions)
    #totals into a list
    total = []
    for i in range(len(conf_m[0])):
        total += [sum(conf_m[i])]
    #Get into percentages
    
    conf_per =  []
    i=0
    for i in range(len(total)):
        line = []
        for j in range(len(total)):
            val = round((conf_m[i][j] / total[i])*100, 1)
            line += [val]
            #print(conf_m[i][j], total[i])
        conf_per += [line]
    print(sum(total))
    print(round(sum(np.diag(conf_per)/3),1))
    
    #class_report = classification_report(y_test, pred[0])
    return conf_m, conf_per

conf_m, conf_per = confusion_table(y_test1, pred1)
print('Confusion Matrix: \n{} \n\nConfusion Percentage: \n{}'.format(conf_m, conf_per))

"""
Question 3
"""
print('\nQuestion 3\n')

times += [time.time()]

#Get new test train data
index2 = (a_idx,b_idx+1)
X_train2, X_test2, y_train2, y_test2 = project(train_data,test_data, eig_vecto, index2)

#Run kNN
nbs=[5] #neighbors
pred2=KNN(X_train2, y_train2, X_test2, nbs) #call the funciton

#Get Accuracy
perc_correct2, perc_wrong2, tot_perc2 = percentage(pred2, fonts, y_test2)
print('Dimension is:', (index2[1]-index2[0])) #dimensions of data
print('Percent Correct: \n{0} \nMean Percent Correct: {1:.2f}'.format(perc_correct2, tot_perc2))

#Get confusion Matrix
conf_m2, conf_per2 = confusion_table(y_test2, pred2)
print(conf_m2, '\n', conf_per2)

"""
Question 4
"""
print('\nQuestion 4\n')

times += [time.time()]

# centroids[i] = [x, y]
def get_centroids(k,ftrs):
    
    import numpy as np
    #starting with random values for centroids
    centroids = {
            i+1: [np.random.randint(-20, 20) for j in range(len(ftrs))]
            for i in range(k)
            }
    #copy the original centroids
    old_centroids = copy.deepcopy(centroids)
    
    return centroids, old_centroids

## ASSIGNMENT
def assignment(df, centroids, lst):
    """
    centroids = new closest centroids
    df = data frame
    lst = features
    """
    for i in centroids.keys():
        # sqrt((x1 - x2)^2 - (y1 - y2)^2)
        # A column for the distance from that point to each centroid
        df['distance_from_{}'.format(i)] = (
            np.sqrt( sum( [ (df[df.columns[j]]-centroids[i][j])**2 
            for j in range(len(lst)) ] ) )
        )
    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
    #the smallest distance gets their index put into the column
    df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
    #df['fonts'] = (int(i[-1]) for i in X_train.loc[:,'closest'])
    return df

## UPDATE
#update the new centroids
def update(centroids, df, lst):
    """
    centroids = new closest centroids
    df = data frame
    lst = features
    """
    #get the cost
    cst=[]
    for i in centroids.keys(): #for every classification
        c=0
        for j in range(len(lst)): #for every feature
            #Get cost before change centroids: magnitude of (x-center)^2
            ct = np.linalg.norm(df[df['closest'] == 'distance_from_{}'.format(i)][lst[j]]-centroids[i][j])
            c += ct**2
        cst += [c]
            
    #update the centroid
    for i in centroids.keys(): #for every classification
        for j in range(len(lst)): #for every feature
            #Get cost before change centroids: magnitude of (x-center)^2
            #Getting the mean of all the values that are closest to that specific class
            #[lst[j]] is calling the column/feature to get the mean of
            centroids[i][j] = np.mean( df[df['closest'] == 'distance_from_{}'.format(i)][lst[j]] )
    
    cost = round(sum(cst),0) 
    #print(cost)
    return centroids, cost, cst

# CONTINUE UNTIL RAN 10 TIMES
X_train_km1 = copy.deepcopy(X_train1)
ftrs1 = X_train1.columns.tolist() # get features as a list 
k = 3

tracker1 = {}
cost_dic1 = {}
for i in range(10):
    centroids1, old_centroids1 = get_centroids(k, ftrs1) #pulls a random #
    for i in range(10): #tries to reduce cost for each random # 10 times
        X_train_km1 = assignment(X_train_km1, centroids1, ftrs1)
        closest_centroids1 = X_train_km1['closest'].copy(deep=True)
        centroids1, cost1, cost_group1 = update(centroids1, X_train_km1, ftrs1)
    #print(cost)
    tracker1[cost1] = centroids1
    cost_dic1[cost1] = cost_group1
    
terminal_costs1 = list(tracker1.keys()) #list of all of the costs
best_cost1 = min(terminal_costs1)
best_centroids1 = tracker1[best_cost1] #pulls the best cost
best_ind_costs1 = cost_dic1[best_cost1] #pulls the individual best costs
X_train_km1 = assignment(X_train_km1, best_centroids1, ftrs1) #make your X_train of your best centroids

c = [] #get column of y values for comparison added to cluster
for i in X_train_km1.loc[:,'closest']:
    c += [int(i[-1])]
y = pd.DataFrame(c) 
y.rename(columns={0:'font'}, inplace=True) #Had to convert to Panda DF and rename column
X_train_km1=pd.concat([X_train_km1,y],axis=1)

"""
Question 5
"""
print('\nQuestion 5\n')

times += [time.time()]

#Get a centroid into each group by splitting and getting mean of each ftr
def known_centroids(df, classification, key):
    """
    df = df of known features and y value (y value must be integer)
    classification: the different classes
    key: column name to compare
    """
    centroids = {}
    cl = {}
    for i in list(classification.keys()):
        group = df.where(df[key] == i) #split into group
        group = group.dropna()
        cl[i] = group[group.columns[1:]]
        group = list(group.mean(axis=0))
        centroids[i] = group[1:] #put that group into dictionary with group as key

    return centroids, cl

#Cost for each cluster from 'ideal'
def get_cost(dictionary, centroids, lst):
    """
    This gets the cost between two dictionary's
    """
    #get the cost
    
    cst=[]
    for i in centroids.keys(): #for every classification
        c=0
        for j in range(len(lst)): #for every feature
            #Get cost before change centroids: (magnitude of (x-center))^2
            ct = np.linalg.norm(dictionary[i][lst[j]]-centroids[i][j])
            c += ct**2
        cst += [round(c,0)]
    
    return cst

XY_train1 = pd.concat([y_train1,X_train1], axis=1) #combine y and x trains for splitting

cl_centroids1, cl_a = known_centroids(XY_train1, fonts, 'font')

cl_cost1 = get_cost(cl_a, cl_centroids1, ftrs1)
cl_total_cost1 = sum(cl_cost1)

#Count all of them that are grouped the same (Confusion Matrix)
km_conf_m1, km_conf_per1 = confusion_table(XY_train1['font'], X_train_km1['font'])
print('Confusion Matrix: \n{} \n\nConfusion Percentage: \n{}'.format(km_conf_m1, km_conf_per1))

times += [time.time()]

"""
Finding centroids for R^(b-a) and ideal centroids then comparing
"""
print('\nExtra\n')

X_train_km2 = copy.deepcopy(X_train2)
ftrs2 = X_train2.columns.tolist() # get features as a list 
k = 3

tracker2 = {}
cost_dic2 = {}
for i in range(10):
    centroids2, old_centroids2 = get_centroids(k, ftrs2) #pulls a random #
    for i in range(10):
        X_train_km2 = assignment(X_train_km2, centroids2, ftrs2)
        closest_centroids2 = X_train_km2['closest'].copy(deep=True)
        centroids2, cost2, cost_group2 = update(centroids2, X_train_km2, ftrs2)
    #print(cost)
    tracker2[cost2] = centroids2
    cost_dic2[cost2] = cost_group2
    
terminal_costs2 = list(tracker2.keys()) #list of all of the costs
best_cost2 = min(terminal_costs2)
best_centroids2 = tracker2[best_cost2] #pulls the best cost
best_ind_costs2 = cost_dic2[best_cost2] #pulls the individual best costs
X_train_km2 = assignment(X_train_km2, best_centroids2, ftrs2) #make your X_train of your best centroids

c = [] #get column of classification values for comparison
for i in X_train_km2.loc[:,'closest']:
    c += [int(i[-1])]
y = pd.DataFrame(c) 
y.rename(columns={0:'font'}, inplace=True) #Had to convert to Panda DF and rename column
X_train_km2=pd.concat([X_train_km2,y],axis=1)

XY_train2 = pd.concat([y_train2,X_train2], axis=1) #combine y and x trains for splitting

cl_centroids2, cl_b = known_centroids(XY_train2, fonts, 'font')

cl_cost2 = get_cost(cl_b, cl_centroids2, ftrs2)
cl_total_cost2 = sum(cl_cost2)

#Count all of them that are grouped the same (Confusion Matrix)
km_conf_m2, km_conf_per2 = confusion_table(XY_train2['font'], X_train_km2['font'])
print('Confusion Matrix: \n{} \n\nConfusion Percentage: \n{}'.format(km_conf_m2, km_conf_per2))

times += [time.time()]
times_c = [i-times[0] for i in times]
print('\n',times_c)