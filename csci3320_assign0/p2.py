import pandas
from scipy.stats import norm
# import matplotlib.pyplot as plt
# import numpy as np
from sklearn import metrics

raw_data = pandas.read_csv("input_2.csv")

for i in range(1,4):
    mean = raw_data[raw_data['class']==i].iloc[:,0].mean()
    var = raw_data[raw_data['class']==i].iloc[:,0].var()
    max = raw_data[raw_data['class']==i].iloc[:,0].max()
    min = raw_data[raw_data['class']==i].iloc[:,0].min()
    print("class = ",i)
    print(len(raw_data[raw_data['class']==i]))
    print("mean = ",mean)
    print("var = ",var)
    print("max = ",max)
    print("min = ",min)
    # raw_data[raw_data['class'] == i]['feature_value'].plot.hist()
    # plt.show()

k = 2400
post = [0,0,0]
x = [0,0,0]
var = [0,0,0]
sd = [0,0,0]

for i in range(3):
    post[i] = len(raw_data.iloc[:k][raw_data.iloc[:k]['class']==i+1])/k
    x[i] = raw_data.iloc[:k][raw_data.iloc[:k]['class']==i+1].iloc[:,0].mean()
    var[i] = raw_data.iloc[:k][raw_data.iloc[:k]['class']==i+1].iloc[:,0].var()
    sd[i] = raw_data.iloc[:k][raw_data.iloc[:k]['class']==i+1].iloc[:,0].std()

print()
print("C = ",post)
print("m = ",x)
print("var = ",var)


def g(alp,i):
    return norm.pdf(alp,loc=x[i],scale=sd[i])*post[i]


#conf_mat = np.zeros((3,3),dtype=int)
label = []
for i in range(k,3000):
    cl = -1
    tmp = 0
    for j in range(3):
        if tmp < g(raw_data.iloc[i,0],j):
            cl = j
            tmp = g(raw_data.iloc[i,0],j)
    #conf_mat[raw_data.iloc[i,1]-1][cl] += 1
    label.append(cl+1)

print("actual row, predict column")
#print(conf_mat)
print(metrics.confusion_matrix(raw_data.iloc[k:,1].to_numpy(),label,labels=[1,2,3]))
print()
print(metrics.classification_report(raw_data.iloc[k:,1],label,labels=[1,2,3],digits=5))