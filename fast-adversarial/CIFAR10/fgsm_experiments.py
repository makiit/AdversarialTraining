import os
import time
for i in range(1,17):
    os.system("python train_fgsm.py --alpha %d --out-dir FGSM_step/alpha_%d"%(i,i))
    time.sleep(3)


# Connect to your background pod via: "kubesh makha22271""
# Please remember to shut down via: "kubectl delete pod makhan-6139" ; "kubectl get pods" to list running pods.
# You may retrieve output from your pod via: "kubectl logs makhan-6139".