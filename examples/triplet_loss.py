from __future__ import print_function, absolute_import
import argparse
import os.path as osp

import numpy as np
import sys
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from reid import datasets
from reid import models
from reid.dist_metric import DistanceMetric
from reid.loss import TripletLoss
from reid.trainers import Trainer
from reid.evaluators import Evaluator
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data.sampler import RandomIdentitySampler
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint


def get_data(name, split_id, data_dir, height, width, batch_size, num_instances, workers, combine_trainval):
    root = osp.join(data_dir, name)

    dataset = datasets.create(name, root, split_id=split_id)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = dataset.trainval if combine_trainval else dataset.train
    num_classes = (dataset.num_trainval_ids if combine_trainval
                   else dataset.num_train_ids)

    train_transformer = T.Compose([
        T.RandomSizedRectCrop(height, width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
    ])

    test_transformer = T.Compose([
        T.RectScale(height, width),
        T.ToTensor(),
        normalizer,
    ])

    train_loader = DataLoader(
        Preprocessor(train_set, root=dataset.images_dir,
                     transform=train_transformer),
        batch_size=batch_size, num_workers=workers,
        sampler=RandomIdentitySampler(train_set, num_instances),
        pin_memory=True, drop_last=True)

    train_loader_head = DataLoader(
        Preprocessor(train_set, root="/home/bfs/zty/reid_market/examples/data/market1501/images_head",
                     transform=train_transformer),
        batch_size=batch_size, num_workers=workers,
        sampler=RandomIdentitySampler(train_set, num_instances),
        pin_memory=True, drop_last=True)
    train_loader_upper = DataLoader(
        Preprocessor(train_set, root="/home/bfs/zty/reid_market/examples/data/market1501/images_upper",
                     transform=train_transformer),
        batch_size=batch_size, num_workers=workers,
        sampler=RandomIdentitySampler(train_set, num_instances),
        pin_memory=True, drop_last=True)
    train_loader_lower = DataLoader(
        Preprocessor(train_set, root="/home/bfs/zty/reid_market/examples/data/market1501/images_lower",
                     transform=train_transformer),
        batch_size=batch_size, num_workers=workers,
        sampler=RandomIdentitySampler(train_set, num_instances),
        pin_memory=True, drop_last=True)

    val_loader = DataLoader(
        Preprocessor(dataset.val, root=dataset.images_dir,
                     transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)
    val_loader_head = DataLoader(
        Preprocessor(dataset.val, root="/home/bfs/zty/reid_market/examples/data/market1501/images_head",
                     transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)
    val_loader_upper = DataLoader(
        Preprocessor(dataset.val, root="/home/bfs/zty/reid_market/examples/data/market1501/images_upper",
                     transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)
    val_loader_lower = DataLoader(
        Preprocessor(dataset.val, root="/home/bfs/zty/reid_market/examples/data/market1501/images_lower",
                     transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    test_loader_head = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root="/home/bfs/zty/reid_market/examples/data/market1501/images_head", transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)
    test_loader_upper = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root="/home/bfs/zty/reid_market/examples/data/market1501/images_upper", transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)
    test_loader_lower = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root="/home/bfs/zty/reid_market/examples/data/market1501/images_lower", transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, num_classes, train_loader, train_loader_head, train_loader_upper, train_loader_lower,\
    val_loader, val_loader_head, val_loader_upper, val_loader_lower, test_loader, test_loader_head, \
    test_loader_upper, test_loader_lower


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    # Redirect print to both console and log file
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))

    # Create data loaders
    assert args.num_instances > 1, "num_instances should be greater than 1"
    assert args.batch_size % args.num_instances == 0, \
        'num_instances should divide batch_size'
    if args.height is None or args.width is None:
        args.height, args.width = (144, 56) if args.arch == 'inception' else \
                                  (256, 128)
    dataset, num_classes, train_loader, train_loader_head, train_loader_upper, train_loader_lower,\
    val_loader, val_loader_head, val_loader_upper, val_loader_lower,\
    test_loader, test_loader_head, test_loader_upper, test_loader_lower= \
        get_data(args.dataset, args.split, args.data_dir, args.height,args.width, args.batch_size,args.num_instances,args.workers,args.combine_trainval)

    # Create model
    # Hacking here to let the classifier be the last feature embedding layer
    # Net structure: avgpool -> FC(1024) -> FC(args.features)
    model = models.create(args.arch, num_features=1024,
                          dropout=args.dropout, num_classes=args.features)
    model_head = models.create(args.arch, num_features=1024,
                          dropout=args.dropout, num_classes=args.features)
    model_upper = models.create(args.arch, num_features=1024,
                          dropout=args.dropout, num_classes=args.features)
    model_lower = models.create(args.arch, num_features=1024,
                          dropout=args.dropout, num_classes=args.features)

    # Load from checkpoint
    start_epoch = best_top1 = 0
#    if args.resume:
#        checkpoint = load_checkpoint(args.resume)
#        model.load_state_dict(checkpoint['state_dict'])
#        start_epoch = checkpoint['epoch']
#       best_top1 = checkpoint['best_top1']
#        print("=> Start epoch {}  best top1 {:.1%}"
#              .format(start_epoch, best_top1))
    model = nn.DataParallel(model).cuda()
    model_head = nn.DataParallel(model_head).cuda()
    model_upper = nn.DataParallel(model_upper).cuda()
    model_lower = nn.DataParallel(model_lower).cuda()

    # Distance metric
    metric = DistanceMetric(algorithm=args.dist_metric)

    # Evaluator
    evaluator = Evaluator(model, model_head, model_upper, model_lower)
    #if args.evaluate:
    #    metric.train(model, train_loader)
    #    print("Validation:")
    #    evaluator.evaluate(val_loader, dataset.val, dataset.val, metric)
    #   print("Test:")
    #    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, metric)
    #    return

    # Criterion
    criterion = TripletLoss(margin=args.margin).cuda()
    criterion_head = TripletLoss(margin=args.margin).cuda()
    criterion_upper = TripletLoss(margin=args.margin).cuda()
    criterion_lower = TripletLoss(margin=args.margin).cuda()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    optimizer_head = torch.optim.Adam(model_head.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay)
    optimizer_upper = torch.optim.Adam(model_upper.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay)
    optimizer_lower = torch.optim.Adam(model_lower.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay)
        

    # Trainer
    trainer = Trainer(model, criterion)
    trainer_head = Trainer(model_head, criterion_head)
    trainer_upper = Trainer(model_upper, criterion_upper)
    trainer_lower = Trainer(model_lower, criterion_lower)

    # Schedule learning rate
    def adjust_lr(epoch):
        lr = args.lr if epoch <= 100 else \
            args.lr * (0.001 ** ((epoch - 100) / 50.0))
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)
            
    def adjust_lr_head(epoch):
        lr = args.lr if epoch <= 100 else \
            args.lr * (0.001 ** ((epoch - 100) / 50.0))
        for g in optimizer_head.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)
            
    def adjust_lr_upper(epoch):
        lr = args.lr if epoch <= 100 else \
            args.lr * (0.001 ** ((epoch - 100) / 50.0))
        for g in optimizer_upper.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)
            
    def adjust_lr_lower(epoch):
        lr = args.lr if epoch <= 100 else \
            args.lr * (0.001 ** ((epoch - 100) / 50.0))
        for g in optimizer_lower.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

    # Start training
    for epoch in range(start_epoch, args.epochs):
        adjust_lr(epoch)
        adjust_lr_head(epoch)
        adjust_lr_upper(epoch)
        adjust_lr_lower(epoch)
        trainer.train(epoch, train_loader, optimizer)
        trainer_head.train(epoch, train_loader_head, optimizer_head)
        trainer_upper.train(epoch, train_loader_upper, optimizer_upper)
        trainer_lower.train(epoch, train_loader_lower, optimizer_lower)
        if epoch < args.start_save:
            continue
        top1 = evaluator.evaluate(val_loader,val_loader_head, val_loader_upper, val_loader_lower, dataset.val, dataset.val)

        is_best = top1 > best_top1
        best_top1 = max(top1, best_top1)
        save_checkpoint({
            'state_dict': model.module.state_dict(),
            'epoch': epoch + 1,
            'best_top1': best_top1,
        }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'), opath= 'model_best.pth.tar')
        
        
        save_checkpoint({
            'state_dict': model_head.module.state_dict(),
            'epoch': epoch + 1,
            'best_top1': best_top1,
        }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint_head.pth.tar'), opath= 'model_head_best.pth.tar')

        save_checkpoint({
            'state_dict': model_upper.module.state_dict(),
            'epoch': epoch + 1,
            'best_top1': best_top1,
        }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint_upper.pth.tar'), opath='model_upper_best.pth.tar')

        save_checkpoint({
            'state_dict': model_lower.module.state_dict(),
            'epoch': epoch + 1,
            'best_top1': best_top1,
        }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint_lower.pth.tar'), opath='model_lower_best.pth.tar')

        print('\n * Finished epoch {:3d}  top1: {:5.1%}  best: {:5.1%}{}\n'.
              format(epoch, top1, best_top1, ' *' if is_best else ''))

    # Final test
    print('Test with best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    checkpoint_head = load_checkpoint(osp.join(args.logs_dir, 'model_head_best.pth.tar'))
    checkpoint_upper = load_checkpoint(osp.join(args.logs_dir, 'model_upper_best.pth.tar'))
    checkpoint_lower = load_checkpoint(osp.join(args.logs_dir, 'model_lower_best.pth.tar'))
    model.module.load_state_dict(checkpoint['state_dict'])
    model_head.module.load_state_dict(checkpoint_head['state_dict'])
    model_upper.module.load_state_dict(checkpoint_upper['state_dict'])
    model_lower.module.load_state_dict(checkpoint_lower['state_dict'])
    metric.train(model, train_loader)
    metric.train(model_head, train_loader_head)
    metric.train(model_upper, train_loader_upper)
    metric.train(model_lower, train_loader_lower)

    evaluator.evaluate(test_loader, test_loader_head, test_loader_upper, test_loader_lower, dataset.query, dataset.gallery, metric)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Triplet loss classification")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='cuhk03',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--height', type=int,
                        help="input height, default: 256 for resnet*, "
                             "144 for inception")
    parser.add_argument('--width', type=int,
                        help="input width, default: 128 for resnet*, "
                             "56 for inception")
    parser.add_argument('--combine-trainval', action='store_true',
                        help="train and val sets together for training, "
                             "val set alone for validation")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 4")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0)
    # loss
    parser.add_argument('--margin', type=float, default=0.5,
                        help="margin of the triplet loss, default: 0.5")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.0002,
                        help="learning rate of all parameters")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--start_save', type=int, default=0,
                        help="start saving checkpoints after specific epoch")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=1)
    # metric learning
    parser.add_argument('--dist-metric', type=str, default='euclidean',
                        choices=['euclidean', 'kissme'])
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    main(parser.parse_args())
