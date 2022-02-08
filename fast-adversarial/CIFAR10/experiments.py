import os
import time
c1 = "python train_pgd.py --alpha 2 --out-dir pgd8_alpha2"
c2 = "python train_pgd.py --alpha 4 --out-dir pgd8_alpha4"
c3 = "python train_pgd.py --restarts 2 --out-dir pgd8_restart2" 
c4 = "python train_pgd.py --opt-level O0 --out-dir pgd8_fp32"
c = [c1,c2,c3,c4]
for x in c:
    print("Running ",x)
    os.system(x)
    time.sleep(3)

