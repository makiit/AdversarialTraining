import os
out_dir = "train_pgd8_alpha_"
alphas = [4,1,0.5]
for a in alphas:
    print("Running experiments for alpha %d"%a)
    command = "python train_pgd.py --alpha %d --out-dir %s"%(a,out_dir+str(a))
    print("Done")
    os.system(command)
