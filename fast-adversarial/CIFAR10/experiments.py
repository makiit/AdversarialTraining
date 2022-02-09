import os
import time
c1 = "python train_pgd.py --alpha 2 --out-dir pgd8_alpha2"
c2 = "python train_pgd.py --alpha 4 --out-dir pgd8_alpha4"
c3 = "python train_pgd.py --restarts 2 --out-dir pgd8_restart2" 
c4 = "python train_pgd.py --opt-level O0 --out-dir pgd8_fp32"
c5 = "python train_pgd.py --epochs 20 --out-dir pgd8_epochs20"
c6 = "python train_pgd.py --epochs 40 --out-dir pgd8_epochs40"
c = [c3,c4,c5,c6]
for x in c:
    print("Running ",x)
    os.system(x)
    time.sleep(3)

# Connect to your background pod via: "kubesh makhan-6139"
# Please remember to shut down via: "kubectl delete pod makhan-6139" ; "kubectl get pods" to list running pods.
# You may retrieve output from your pod via: "kubectl logs makhan-6139".