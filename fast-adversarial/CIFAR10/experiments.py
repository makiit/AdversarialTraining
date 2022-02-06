import os
out_dir = "train_pgd_"
for i in range(7,10):
    print("Running experiments for epsilon %d"%i)
    command = "python train_pgd.py --epsilon %d --out-dir %s"%(i,out_dir+str(i))
    os.system(command)
