python train_pgd.py --alpha 2 --out-dir pgd8_alpha2 --epoch 20
python train_pgd.py --alpha 4 --out-dir pgd8_alpha4 --epoch 20
python train_pgd.py --restarts 2 --out-dir pgd8_restart2 
python train_pgd.py --opt-level O0 --out-dir pgd8_fp32

