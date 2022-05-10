import argparse
import logging
import os
import time

# import apex.amp as amp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from preact_resnet import PreActResNet18
from utils_normal import (upper_limit, lower_limit, std, clamp, get_loaders,
    evaluate_pgd, evaluate_standard)

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='../../cifar-data', type=str)
    parser.add_argument('--root',type=str,required=True)
    return parser.parse_args()


def main():
    args = get_args()

    if not os.path.exists(args.root):
        os.mkdir(args.root)
    logfile = os.path.join(args.root, 'output_pgd.log')
    if os.path.exists(logfile):
        os.remove(logfile)

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=logfile)
    logger.info(args)

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    cpts = os.listdir(args.root)
    cpts = [x for x in cpts if ".pth" in x]
    def getEpoch(s):
        t = s.split('.')
        t = t[0].split('_')
        epoch = t[-1]
        return int(epoch)

    def compare(s1,s2):
        e1=getEpoch(s1)
        e2=getEpoch(s2)
        if(e1<e2):return -1
        else:return 1


    cpts=sorted(cpts, key=cmp_to_key(compare))
    cpts = [args.root+x for x in cpts]

    for cpt in cpts:
        ckpt = torch.load(cpt)

        train_loader, test_loader = get_loaders(args.data_dir, args.batch_size)
        model_test = PreActResNet18().cuda()
        

        state_dict = model_test.state_dict()
        for k in ckpt_copy:
            if(k not in state_dict):
                del ckpt[k]
        model_test.load_state_dict(ckpt)
        model_test.float()
        model_test.eval()

        pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, 50, 10)
        test_loss, test_acc = evaluate_standard(test_loader, model_test)
        print("Checkpoint %s PGD Accuracy %f"%cpt,pgd_acc)
        logger.info('Checkpoint \t Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
        logger.info('%s \t %.4f \t \t %.4f \t %.4f \t %.4f',cpt, test_loss, test_acc, pgd_loss, pgd_acc)




if __name__ == "__main__":
    main()
