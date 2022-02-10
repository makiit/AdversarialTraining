import matplotlib.pyplot as plt 
import numpy as np 
import argparse
import re

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str)
    return parser.parse_args()

args = get_args()
f = open(args.file,"r")
pgd_acc = []
train_acc = []
lines = f.readlines()
for i in range(len(lines)):
    if("Epoch" in lines[i] and "PGD" in lines[i]):
        print("Test ",lines[i+1])
        print("Train ",lines[i+2])
        x = lines[i+1].split(' ')[-1]
        y = lines[i+2].split(' ')[-1]
        pgd_acc.append((1-float(x[:-1]))*100)
        train_acc.append((1-float(y[:-1]))*100)
        i+=2
print(pgd_acc)
print(train_acc)
plt.plot(pgd_acc,c="royalblue",label = "PGD Test",linewidth=3.0)
plt.plot(train_acc,c="darkorange",label = "FGSM Train",linewidth=3.0)
plt.xlabel("Epochs")
plt.ylabel("% Error")
plt.legend()
plt.savefig("Overfitting_our.png")
plt.show()




