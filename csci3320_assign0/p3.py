import pandas
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn.metrics as metrics

raw_data = pandas.read_csv("input_3.csv")
print(raw_data)
k = math.floor(len(raw_data)*0.8)
train = raw_data.iloc[:k]
test = raw_data.iloc[k:]

print()

post = np.zeros((3),dtype=int)
var = [0,0,0]
sd = np.zeros((3,2))
mean = [0,0,0]
calc_mean = np.zeros((3,2))

for i in range(1,4):
    print("class = ",i)
    tmp = raw_data[raw_data['class']==i]
    print(len(tmp))
    print("mean x_1 = ",tmp['feature_value_1'].mean())
    print("mean x_2 = ",tmp['feature_value_2'].mean())
    print("var x_2 = ",tmp['feature_value_2'].var())
    tmp['feature_value_2'].plot.hist()
    # plt.show()
    for j in range(2):
        print("mean x_2 | x_1=",j," = ",tmp[tmp['feature_value_1']==j]['feature_value_2'].mean())
        print("var x_2 | x_1=",j," = ",tmp[tmp['feature_value_1']==j]['feature_value_2'].var())
        tmp[tmp['feature_value_1'] == j]['feature_value_2'].plot.hist()
        # plt.show()
    print()

for i in range(3):
    tmp = train[train['class'] == i + 1]
    post[i] = len(tmp)
    mean[i] = tmp.iloc[:,:2].mean()
    var[i] = tmp.iloc[:,:2].var()
    for j in range(2):
        calc_mean[i][j] = tmp[tmp.iloc[:,0]==j].iloc[:,1].mean()
        sd[i][j] = tmp[tmp.iloc[:,0]==j].iloc[:,1].std()


def g(x1, x2, i):
    p = mean[i].to_numpy()[0]
    a = stats.norm.pdf(x2, loc=calc_mean[i][j], scale=sd[i][j])
    a = math.log(a)
    b = x1 * p + (1 - x1) * (1 - p)
    b = math.log(b)
    c = post[i]
    c = math.log(c)
    return a + b + c


def g2(x1,x2,i):
    return math.log(stats.multivariate_normal(mean=mean[i].to_numpy(),cov=var[i].to_numpy()).pdf(np.array([x1,x2]))) + math.log(post[i])


label = []
for i in range(3000-k):
    cl = 2
    dat = test.iloc[i].to_numpy()
    prob = g(dat[0],dat[1],2)
    for j in range(2):
        if g(dat[0],dat[1],j) > prob:
            prob = g(dat[0],dat[1],j)
            cl = j
    label.append(cl+1)

post = pow(k,-1)*post
print("C = ",post)
print("mean = ",mean)
print("variance = ",var)

print(metrics.confusion_matrix(test.iloc[:,2].to_numpy(),label,labels=[1,2,3]))
print(metrics.classification_report(test.iloc[:,2].to_numpy(),label,labels=[1,2,3],digits=5))