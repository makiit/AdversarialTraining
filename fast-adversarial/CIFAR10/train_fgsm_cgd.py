import argparse
import logging
import os
import time
from CGDs import ACGD, BCGD
import apex.amp as amp
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

from preact_resnet import PreActResNet18
from utils import (upper_limit, lower_limit, std, clamp, get_loaders,
    evaluate_pgd, evaluate_standard)

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--data-dir', default='../../cifar-data', type=str)
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--lr-schedule', default='multistep', type=str, choices=['cyclic', 'multistep'])
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.2, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--cgd-iter', default=1, type=int, help='Attack iterations')
    parser.add_argument('--restarts', default=1, type=int)
    parser.add_argument('--alpha', default=2, type=int, help='Step size')
    parser.add_argument('--delta-init', default='random', choices=['zero', 'random'],
        help='Perturbation initialization method')
    parser.add_argument('--out-dir', default='CGD3', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--opt-level', default='O2', type=str, choices=['O0', 'O1', 'O2'],
        help='O0 is FP32 training, O1 is Mixed Precision, and O2 is "Almost FP16" Mixed Precision')
    parser.add_argument('--loss-scale', default='1.0', type=str, choices=['1.0', 'dynamic'],
        help='If loss_scale is "dynamic", adaptively adjust the loss scale over time')
    parser.add_argument('--master-weights', action='store_true',
        help='Maintain FP32 master weights to accompany any FP16 model weights, not applicable for O1 opt level')
    parser.add_argument('--eval',default=False)
    parser.add_argument('--test-interval',default=3,type=int)
    return parser.parse_args()


def main():
    args = get_args()

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    logfile = os.path.join(args.out_dir, 'output.log')
    if os.path.exists(logfile):
        os.remove(logfile)

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=logfile)
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size)

    epsilon = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std
    if(args.eval==False):
        model = PreActResNet18().cuda()
        model.train()
        delta = torch.zeros([args.batch_size,3,32,32], requires_grad=True)
        # opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)
        # amp_args = dict(opt_level=args.opt_level, loss_scale=args.loss_scale, verbosity=False)
        # if args.opt_level == 'O2':
        #     amp_args['master_weights'] = args.master_weights
        # model, opt = amp.initialize(model, opt, **amp_args)
        criterion = nn.CrossEntropyLoss()
        lr_steps = args.epochs * len(train_loader)
        # if args.lr_schedule == 'cyclic':
        #     scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
        #         step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
        # elif args.lr_schedule == 'multistep':
        #     scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)

        # Training
        start_train_time = time.time()
        logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc \t Delta norm')
        for epoch in range(args.epochs):
            start_epoch_time = time.time()
            train_loss = 0
            train_acc = 0
            norm_delta = 0
            train_n = 0
            i = 0
            with tqdm(train_loader, unit="batch") as tepoch:
                for X, y in tepoch:
                    tepoch.set_description("Epoch %d"%epoch)
                    # print("Epoch %d Iteration %d"%(epoch,i))
                    X, y = X.cuda(), y.cuda()
                    delta = torch.zeros_like(X).cuda()
                    
                    if args.delta_init == 'random':
                        for i in range(len(epsilon)):
                            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
                        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
                    delta.requires_grad = True
                    opt = BCGD(max_params=[delta],min_params=model.parameters(),lr_max = 0.2,lr_min = 0.2)
                    for ci in range(args.cgd_iter):
                        output = model(X + delta)
                        loss = criterion(output, y)
                        opt.step(loss=loss)
                        delta.data = clamp(delta, -epsilon, epsilon)
                        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
                    train_loss += loss.item() * y.size(0)
                    train_acc += (output.max(1)[1] == y).sum().item()
                    norm_delta += torch.mean(torch.norm(delta,p=float('inf'),dim = [1,2,3])).item()
                    train_n += y.size(0)

                    tepoch.set_postfix(loss=loss.item()/train_n, accuracy=100. * train_acc/train_n)
                    # sleep(0.01)
                    # scheduler.step()
                    i+=1
            if(epoch % args.test_interval==0 and epoch>0):
                model.eval()
                pgd_loss, pgd_acc = evaluate_pgd(test_loader, model, 25, 10)
                test_loss, test_acc = evaluate_standard(test_loader, model)
                model.train()
                logger.info('Epoch \t Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
                logger.info('%d \t %.4f \t \t %.4f \t %.4f \t %.4f',epoch,test_loss, test_acc, pgd_loss, pgd_acc)
            epoch_time = time.time()
            logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f',
                epoch, epoch_time - start_epoch_time,train_loss/train_n, train_acc/train_n,norm_delta/train_n)
        train_time = time.time()
        torch.save(model.state_dict(), os.path.join(args.out_dir, 'model.pth'))
        logger.info('Total train time: %.4f minutes', (train_time - start_train_time)/60)

    # Evaluation

    model_test = PreActResNet18().cuda()
    if(args.eval==True):
        torch.load(model_test,os.path.join(args.out_dir, 'model.pth'))


if __name__ == "__main__":
    main()