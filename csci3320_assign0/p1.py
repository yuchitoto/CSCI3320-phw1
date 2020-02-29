import pandas
import numpy
from sklearn import metrics
import math

raw_data = pandas.read_csv("input_1.csv")
for i in range(1,4):
    mean = raw_data[raw_data['class']==i].iloc[:,0].mean()
    var = raw_data[raw_data['class']==i].iloc[:,0].var()
    max = raw_data[raw_data['class']==i].iloc[:,0].max()
    min = raw_data[raw_data['class']==i].iloc[:,0].min()
    print("class = ",i)
    print("mean = ",mean)
    print("var = ",var)
    print("max = ",max)
    print("min = ",min)
# traindata, testdata = sel.train_test_split(raw_data, test_size=0.2)
acc = [0,0,0]
ones = [0,0,0]
k = 300*8
print("p = ",k)
p = raw_data.iloc[:,0].mean()
print(p)
for i in range(k):
    if raw_data.iloc[i,1] == 1:
        acc[0] += 1
        ones[0] += raw_data.iloc[i,0]
    if raw_data.iloc[i,1] == 2:
        acc[1] += 1
        ones[1] += raw_data.iloc[i,0]
    if raw_data.iloc[i,1] == 3:
        acc[2] += 1
        ones[2] += raw_data.iloc[i,0]
print("P(C1) = ",acc[0]/k)
print("P(C2) = ",acc[1]/k)
print("P(C3) = ",acc[2]/k)
print("p1 = ",ones[0]/acc[0])
print("p2 = ",ones[1]/acc[1])
print("p3 = ",ones[2]/acc[2])


def g(a, b):
    return math.log(a) + math.log(b)


g1 = g(ones[0]/acc[0],acc[0]/k)
g2 = g(ones[1]/acc[1],acc[1]/k)
g3 = g(ones[2]/acc[2],acc[2]/k)
print(g1)
print(g2)
print(g3)

conf_mat = numpy.zeros((3,3),dtype=int)

precision = [0,0,0]
accuracy = 0
recall = [0,0,0]
f1 = [0,0,0]

label = []
for i in range(k,3000):
    cl = 4
    if raw_data.iloc[i,0] == 0:
        g1 = g(1-ones[0]/acc[0],acc[0]/k)
        g2 = g(1-ones[1] / acc[1], acc[1] / k)
        g3 = g(1-ones[2] / acc[2], acc[2] / k)
    if raw_data.iloc[i,0] == 1:
        g1 = g(ones[0] / acc[0], acc[0] / k)
        g2 = g(ones[1] / acc[1], acc[1] / k)
        g3 = g(ones[2] / acc[2], acc[2] / k)
    if g1>g2 and g1>g3:
        cl = 0
    if g2>g1 and g2>g3:
        cl = 1
    if g3>g1 and g3>g2:
        cl = 2
    conf_mat[raw_data.iloc[i,1]-1][cl] += 1
    label.append(cl+1)

print("actual row, predict column")
#print(conf_mat)
print(metrics.confusion_matrix(raw_data.iloc[k:,1].to_numpy(),label,labels=[1,2,3]))
print()
print(metrics.classification_report(raw_data.iloc[k:,1],label,labels=[1,2,3],digits=5,zero_division=1))
print("actual row, predict column")
print(numpy.array(conf_mat))
tpfp = numpy.array(conf_mat).sum(axis=0)
tpfn = numpy.array(conf_mat).sum(axis=1)
print(tpfp)
print(tpfn)

for i in range(3):
    accuracy += conf_mat[i][i]
    precision[i] = conf_mat[i][i]/tpfp[i]
    recall[i] = conf_mat[i][i]/tpfn[i]
    f1[i] = 2*(precision[i]*recall[i])/(precision[i]+recall[i])

accuracy /= 3000-2400
print("accuracy = ",accuracy)
print("precision = ",precision)
print("recall = ",recall)
print("f1 = ",f1)
