import argparse
import random
import torch
import time
import datetime
import sys
import math

import torchvision.models as models
import custom_models as ctm_models
import torch.optim.lr_scheduler as lr_scheduler

from datasets.customs import \
    SSSg, _convert_torch_to_pil, MyWrapperLoader, INaturalist, ResizeAPad, SCL, prior_label, os, np, nn, \
    transforms, _convert_channel, sample_assignment, My2WrapperLoader
from datasets.transfer import build_transfer
from datasets.results import result_compute
from log.utils import AverageMeter
from torch.utils.data import DataLoader
from torchvision.models._utils import IntermediateLayerGetter
from torch import optim
from torch.nn import functional as F
from tensorboardX import SummaryWriter
from optimizer.random_fourier import loss_expect, Variable, einops


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='datasets/sss', metavar='DIR')  #
    parser.add_argument('--root_log', type=str, default='log')
    parser.add_argument('--root_weights', type=str, default='checkpoints')
    parser.add_argument('--arch', default='resnet50', choices=['resnet50', 'resnet101'])
    parser.add_argument('--batch', default=64, type=int)
    parser.add_argument('--img_size', default=224, type=int, help='image sizes')
    parser.add_argument('--m_lr', default=0.001, type=float, help='learning rate for backbone model')
    parser.add_argument('--c_lr', default=0.002, type=float, help='learning rate for classifier')
    parser.add_argument('--ratio_prior', default=100., type=float, help='ratio of seen-unseen prior')
    parser.add_argument('--model_transfer', type=str, default='ccpl', help='photowct, liwct, stytr, ccpl')
    parser.add_argument('--mode_transfer', type=str, default='none')
    parser.add_argument('--std_multi', default=0.01, type=float,    # 0.01
                        help='standard deviation of multiple noise added on sonar images')
    parser.add_argument('--std_plus', default=0.1, type=float,      # 0.1
                        help='standard deviation of guass noise added on sonar images')
    parser.add_argument('--std_norm', default=((0.3628, 0.3628, 0.3643), (0.1500, 0.1500, 0.1505)),
                        type=tuple, help='normalization for sonar images')
    parser.add_argument('--mfa_flag', default=False, type=bool, help='feature adaptation in intermedatie layer')
    parser.add_argument('--cfa_flag', default=False, type=bool, help='causal feature adaptation')
    parser.add_argument('--alpha1', default=1., type=float, help='global loss for mid-features of module 2')
    parser.add_argument('--alpha2', default=1., type=float, help='laplace loss for mid-features of module 2')
    parser.add_argument('--beta1', default=1., type=float, help='global loss for mid-features of module 3')
    parser.add_argument('--beta2', default=1., type=float, help='laplace loss for mid-features of module 3')
    parser.add_argument('--gamma', default=1., type=float, help='ce loss')
    parser.add_argument('--workers', default=0, type=int)
    parser.add_argument('--seed', default=1, type=int, help='seed for initializing training')
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--epoch', default=10, type=int)        #
    parser.add_argument('--config', dest='config', help='settings of sonar image classifying in yaml format')
    parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
    args = parser.parse_args()

    return args


def laplace_operator(m1, m2, k):
    return F.conv2d(m1, k), F.conv2d(m2, k)


def train(train_loader, model_bone, optimizer, scheduler, args, epoch, tf_writer):
    batch_time = AverageMeter('Time', ':6.3f')

    semantic_dict1 = {}
    semantic_dict1.clear()
    semantic_dict2 = {}
    semantic_dict2.clear()
    feature_dict1 = {}
    feature_dict1.clear()
    feature_dict2 = {}
    feature_dict2.clear()

    model_bone.train()
    end = time.time()
    scheduler.step(epoch)
    for i, data in enumerate(train_loader):
        input1, input2, targets = data
        input1, input2, targets = input1.to(args.device), input2.to(args.device), targets.to(args.device)
        feat1 = model_bone(input1) 
        feat2 = model_bone(input2)
        label_list = torch.unique(targets).cpu().tolist()
        w_hist = torch.tensor([(targets == i).sum() for i in range(4)]).to(args.device)[targets]
        w_hist = 1. / w_hist

        # Cross entropy with Long-Tail Weights
        feat1 = feat1 + args.log_p0_y.to(args.device) 
        feat2 = feat2 + args.log_p0_y.to(args.device) 
        loss_ce = F.cross_entropy(feat1, targets) + F.cross_entropy(feat2, targets)   
        loss_kl = F.kl_div(F.softmax(feat1, dim=1).log(), F.softmax(feat2, dim=1), reduction='mean') + \
            F.kl_div(F.softmax(feat2, dim=1).log(), F.softmax(feat1, dim=1), reduction='mean')
        loss = loss_ce + loss_kl * (1 - epoch / args.epoch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] " 
                         "[total loss: %.4f; ]" "[batch_time: %.4f(s)]"
                         % (epoch, args.epoch, i, len(train_loader), loss.item(),
                            # loss_ce.item(), loss1.item(), loss2.item(),
                            time.time() - end))
        end = time.time()

        semantic_dict1 = {}
        semantic_dict1.clear()
        semantic_dict2 = {}
        semantic_dict2.clear()
        feature_dict1 = {}
        feature_dict1.clear()
        feature_dict2 = {}
        feature_dict2.clear()

    print()
    torch.cuda.empty_cache()
    return print("---------- {}-epoch training completed ----------".format(epoch))


def val(loader, model, num_classes, args, epoch=0):
    model.eval()
    num_sample = loader.dataset.root_path.__len__() * 4
    prob_val = torch.FloatTensor(num_sample, num_classes)
    pred_val = torch.LongTensor(num_sample)
    label_val = torch.LongTensor(num_sample)
    time_end = time.time()
    with torch.no_grad():
        start = 0
        for i, data in enumerate(loader):
            inputs, targets = data
            end = min(num_sample, start + targets.shape[0])
            inputs = inputs.to(args.device)
            logits = model(inputs)
            probs = F.softmax(logits.detach().cpu(), dim=-1)
            prob_val[start:end] = probs
            pred_val[start:end] = torch.max(probs.data, 1)[1].cpu()
            label_val[start:end] = targets
            start = end

    batch_time = time.time() - time_end
    sys.stdout.write("\r[Epoch %d/%d] " "\r[testing_time: %.4f(s)]" % (epoch, args.epoch, batch_time))
    print()
    torch.cuda.empty_cache()

    return prob_val, pred_val, label_val


def main():
    # Load config file
    args = get_arguments()
    current_time = datetime.datetime.now()
    args.store_name = ''.join(
        ['Bi_ctr(a1-p2)_', str(args.model_transfer), '-Net_', args.arch, '-Batchsize', str(args.batch),    # Bi_ctr_ My_ Bi_ My(a4-p2)_
         '-Mid_lr', str(args.m_lr), '-lr', str(args.c_lr), '-Ratio', str(int(args.ratio_prior)),
         '-Mode', str(args.mode_transfer)]) + '-Time' + current_time.strftime("%Y%m%d%H%M%S")

    datasets = SSSg(folder=args.data)
    device = torch.device("cuda") if args.cuda and torch.cuda.is_available() else torch.device("cpu")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    style_model = build_transfer(args.model_transfer, mode=args.mode_transfer).to(device)
    for p in style_model.parameters():
        p.requires_grad = False

    style_size = 224 if args.model_transfer in ['stytr', 'gcea'] else 512 if args.model_transfer in ['asepa'] else 256
    transform_train_s1 = [
        SCL(args.img_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Compose([
            transforms.Resize((style_size, style_size)),
            transforms.ToTensor(),
        ])
    ]
    transform_train_s2 = transforms.Compose([
        _convert_torch_to_pil,
        transforms.Resize(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(args.std_norm[0], args.std_norm[1])
    ])
    transform_test = transforms.Compose([
        ResizeAPad(args.img_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        _convert_channel,
        transforms.Normalize(args.std_norm[0], args.std_norm[1])
    ])

    hist_sampler = np.array([1 if i in datasets.unseenclasses else
                             math.ceil((args.batch - 1 * datasets.unseenclasses.size) / datasets.seenclasses.size)
                             for i in range(datasets.num_classes)])
    train_wrapper = MyWrapperLoader(style_model, datasets.train_x, datasets.classnames, hist_sampler,   # MyWrapperLoader
                                    trans_s1=transform_train_s1, trans_s2=transform_train_s2,
                                    in_seen=datasets.seenclasses, in_unseen=datasets.unseenclasses,
                                    flag=(args.model_transfer, args.mode_transfer),
                                    std_n=(args.std_multi, args.std_plus), device=device)
    train_loader = DataLoader(train_wrapper, batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, pin_memory=torch.cuda.is_available())
    test_wrapper = INaturalist(datasets.test, trans=transform_test)
    test_loader = DataLoader(test_wrapper, batch_size=100, shuffle=False,
                             num_workers=args.workers, pin_memory=torch.cuda.is_available())

    custom_model = getattr(ctm_models, args.arch + '_m')(pretrained=True)   # True

    for name, param in custom_model.named_parameters():
        if name.split('.')[0] not in ['layer1', 'layer2', 'layer3', 'layer4', 'fc']:
            param.requires_grad = False

    n_classes = datasets.train_x.__len__()
    custom_model.fc = nn.Linear(custom_model.fc.in_features, n_classes)
    args.device = torch.device("cuda") if args.cuda and torch.cuda.is_available() else torch.device("cpu")
    custom_model.to(args.device)

    optimizer = optim.SGD([
        # {'params': custom_model.conv1.parameters(), 'lr': args.m_lr},
        # {'params': custom_model.bn1.parameters(), 'lr': args.m_lr},
        {'params': custom_model.layer1.parameters(), 'lr': args.m_lr},
        {'params': custom_model.layer2.parameters(), 'lr': args.m_lr},
        {'params': custom_model.layer3.parameters(), 'lr': args.m_lr},
        {'params': custom_model.layer4.parameters(), 'lr': args.m_lr},
        {'params': custom_model.fc.parameters()}], lr=args.c_lr, momentum=0.9, weight_decay=0.05)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True)

    args.hist = torch.from_numpy(hist_sampler).to(device)   #
    args.log_p_y, args.log_p0_y = \
        prior_label(hist_sampler.astype(float), datasets.seenclasses, datasets.unseenclasses, args.ratio_prior, device)
    args.laplac_k = torch.tensor([[-0.707, -1., -0.707],
                                  [-1, 7.828, -1],
                                  [-0.707, -1, -0.707]], dtype=torch.float).view(1, 1, 3, 3).to(device)
    args.laplac_k = args.laplac_k / args.laplac_k.sum()
    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))
    for epoch in range(args.start_epoch, args.epoch):
        train(train_loader, custom_model, optimizer, scheduler, args, epoch, tf_writer)
        output, pred, target = val(test_loader, custom_model, n_classes, args, epoch=epoch)
        cfs_mat, accuracy_list, g_mean, macro_f1, roc_list, ccr_macro_list, h_macro_list = \
            result_compute(output, pred, target, datasets.classnames, s=datasets.seenclasses, u=datasets.unseenclasses)

        sys.stdout.write("\rEpoch %d Results -- " % epoch)
        print()
        sys.stdout.write("\r1. Confusion Matrix: ")
        print(cfs_mat)
        sys.stdout.write("\r2. Global Acc: %.2f, Mean Acc: %.2f, Tr: %.2f, Ts: %.2f, H: %.2f" % (
            accuracy_list[0], accuracy_list[1], accuracy_list[2], accuracy_list[3], accuracy_list[4]))
        print()
        sys.stdout.write("\r3. G_mean: ")
        print(g_mean)
        sys.stdout.write("\r4. Macro−F1-score: %.2f" % (macro_f1 * 100.))
        print()
        sys.stdout.write("\r5. ROC curve integral-area: ")
        print(roc_list[2])
        sys.stdout.write("\r6. Macro-average CCR macro-FPR curve integral-area: %.2f" % (ccr_macro_list[2] * 100.))
        print()
        sys.stdout.write("\r7. harmonic CCR macro-FPR curve integral-area: %.2f" % (h_macro_list[2] * 100.))
        print()


if __name__ == '__main__':
    main()


