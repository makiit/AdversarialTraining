import os
import time
c1 = "python train_fgsm.py --delta-init zero --out-dir FGSM_DB/zero_init"
c2 = "python train_fgsm.py --early-stop  --delta-init zero --out-dir FGSM_DB/zero_init_early"
c3 = "python train_fgsm.py --delta-init previous --out-dir FGSM_DB/prev_init"
c4 = "python train_fgsm.py --delta-init random --alpha 8 --out-dir FGSM_DB/random_init_alpha8"
c5 = "python train_fgsm.py --delta-init random  --out-dir FGSM_DB/random_init_alpha10"
c6 = "python train_fgsm.py --delta-init random --alpha 16 --out-dir FGSM_DB/random_init_alpha16"
c7 = "python train_fgsm.py --early-stop --delta-init random --alpha 16 --out-dir FGSM_DB/random_init_alpha16_early"
c8 = "python train_fgsm.py --delta-init zero --alpha 8 --test-interval 1 --out-dir FGSM_DB/zero_init_plot"
c = [c2,c5,c6,c7,c8,c4]
for x in c:
    print("Running ",x)
    os.system(x)
    time.sleep(3)

# Connect to your background pod via: "kubesh makha22271""
# Please remember to shut down via: "kubectl delete pod makhan-6139" ; "kubectl get pods" to list running pods.
# You may retrieve output from your pod via: "kubectl logs makhan-6139".