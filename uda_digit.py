import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import network, loss
from torch.utils.data import DataLoader
import random, pdb, math, copy
from scipy.spatial.distance import cdist
from DANN_all_data import load_data_subject


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


# def load_seed_data(name, folder):
#     seed_path = os.path.join(folder, name+".mat")
#     data = scio.loadmat(seed_path)
#     label_path = os.path.join(folder, "label.mat")
#     label = scio.loadmat(label_path)
#     data_list=[]
#     label_list=[]
#
#     # print(data.keys())
#     for index in data.keys():
#         i = index[-2:]
#         if i.isdigit():
#             i = int(i)
#         elif i[-1].isdigit():
#             i = int(i[-1])
#         else:
#             continue
#         data_list.append(data[index].transpose(1, 0, 2))
#         label_list = label_list+[label['label'][0][i-1]]*data[index].shape[1]
#     source_data = np.concatenate(data_list)
#     label = np.array(label_list)
#     val0 = np.mean(source_data[:, :, 0])
#     val1 = np.mean(source_data[:, :, 1])
#     val2 = np.mean(source_data[:, :, 2])
#     val3 = np.mean(source_data[:, :, 3])
#     val4 = np.mean(source_data[:, :, 4])
#
#     source_data[:, :, 0] = source_data[:, :, 0] - val0  ### 标准化的减去均值
#     source_data[:, :, 1] = source_data[:, :, 1] - val1
#     source_data[:, :, 2] = source_data[:, :, 2] - val2
#     source_data[:, :, 3] = source_data[:, :, 3] - val3
#     source_data[:, :, 4] = source_data[:, :, 4] - val4
#
#     source_data[:, :, 0] = 2 * source_data[:, :, 0] / val0  ###  两倍内容除以均值
#     source_data[:, :, 1] = 2 * source_data[:, :, 1] / val1
#     source_data[:, :, 2] = 2 * source_data[:, :, 2] / val2
#     source_data[:, :, 3] = 2 * source_data[:, :, 3] / val3
#     source_data[:, :, 4] = 2 * source_data[:, :, 4] / val4
#
#     return source_data, label

class MyDataset():
    def __init__(self, data, labels):
        self.eeg = data
        self.labels = labels

    def __getitem__(self, index):
        img, target = self.eeg[index], self.labels[index]
        return img, target

    def __len__(self):
        return len(self.eeg)

# class Source_Dataset():
#     def __init__(self, data, labels, domain_label):
#         self.eeg = data
#         self.labels = labels
#         self.domain_label = domain_label
#
#     def __getitem__(self, index):
#         img, target, domain_label = self.eeg[index], self.labels[index], self.domain_label[index]
#         return img, target, domain_label
#
#     def __len__(self):
#         return len(self.eeg)

# def digit_load(args):
#     folder = "/home/ydwang/wangDataDisk/seed/"
#     name_source = "DE_LDS_dujingcheng_20131027"
#     name_target = "DE_LDS_jianglin_20140413"
#     train_bs = 30
#     source_data, source_label = load_seed_data(name_source, folder)
#     print(source_data.shape)
#     print(source_label.shape)
#
#
#     target_data, target_label = load_seed_data(name_target, folder)
#     print(target_data.shape)
#     print(target_label.shape)
#
#     train_source = MyDataset(source_data, source_label)
#     dset_loaders = {}
#     dset_loaders["source_tr"] = DataLoader(train_source, batch_size=train_bs, shuffle=True,
#         num_workers=args.worker, drop_last=False)
#     dset_loaders["source_te"] = DataLoader(train_source, batch_size=train_bs*2, shuffle=True,
#         num_workers=args.worker, drop_last=False)
#
#     train_target = MyDataset(target_data, target_label)
#     test_target = train_target
#     dset_loaders["target"] = DataLoader(train_target, batch_size=train_bs, shuffle=True,
#         num_workers=args.worker, drop_last=False)
#     dset_loaders["target_te"] = DataLoader(train_target, batch_size=train_bs, shuffle=False,  ##
#         num_workers=args.worker, drop_last=False)
#     dset_loaders["test"] = DataLoader(test_target, batch_size=train_bs*2, shuffle=False,
#         num_workers=args.worker, drop_last=False)
#     return dset_loaders



def cal_acc2(loader, netF, netC):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            # print(labels.shape) # 128 1
            inputs = inputs.cuda()
            # outputs = netC(netB(netF(inputs)))

            outputs = netC(netF(inputs))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = (labels).float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, (labels).float()), 0)
    _, predict = torch.max(all_output, 1)
    # print(predict)
    # print(all_label)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label.T).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()
    return accuracy*100, mean_ent





import torch.nn.functional as F
def DMI_loss(output, target):
    outputs = F.softmax(output, dim=1)
    outputs = outputs
    # print(torch.typename(outputs))
    targets = target.reshape(target.size(0), 1).cpu()
    y_onehot = np.zeros((target.size(0), 3))
    y_onehot = torch.Tensor(y_onehot)
    y_onehot.scatter_(1, targets, 1)
    # y_onehot = y_onehot.cpu()

    y_onehot = y_onehot.t()
    # outputs = torch.Tensor(outputs)
    y_onehot = y_onehot.float().cuda()
    # y_onehot = torch.Tensor(y_onehot)
    # print(y_onehot.shape)
    # print(outputs.shape)
    # print(torch.typename(y_onehot))
    mat = torch.mm(y_onehot, outputs)
    # mat = mat / target.size(0)
    return -1.0 * torch.log(torch.abs(torch.det(mat.float())) + 0.001)

def digit_load(args):
    index = args.index
    train_bs = 128
    source_data, source_label, target_data, target_label, domain_label = load_data_subject(index)
    print("source_data", source_data.shape)
    print("source_label", source_label.shape)
    print("domain_label", domain_label.shape)
    print("target_data", target_data.shape)
    print("target_label", target_label.shape)

    train_source = MyDataset(source_data, source_label)
    domian_source = MyDataset(source_data, domain_label)
    domian_test = MyDataset(source_data[3000:6000], domain_label[3000:6000])

    dset_loaders = {}
    dset_loaders["source_tr"] = DataLoader(train_source, batch_size=train_bs, shuffle=True,
        num_workers=args.worker, drop_last=False)

    dset_loaders["source_domain"] = DataLoader(domian_source, batch_size = train_bs, shuffle=True,
                                               num_workers=args.worker, drop_last=False)
    middle = 3000

    train_target = MyDataset(target_data[:middle], target_label[:middle])
    test_target = MyDataset(target_data[middle:], target_label[middle:])
    dset_loaders["source_te"] = DataLoader(train_target, batch_size=train_bs, shuffle=True,
        num_workers=args.worker, drop_last=True)
    dset_loaders["domain_te"] = DataLoader(domian_test, batch_size=train_bs, shuffle=True,
        num_workers=args.worker, drop_last=True)
    dset_loaders["target"] = DataLoader(train_target, batch_size=train_bs, shuffle=True,
        num_workers=args.worker, drop_last=False)
    dset_loaders["target_te"] = DataLoader(train_target, batch_size=train_bs, shuffle=True,  ##
        num_workers=args.worker, drop_last=False)
    dset_loaders["test"] = DataLoader(test_target, batch_size=train_bs, shuffle=True,
        num_workers=args.worker, drop_last=False)
    return dset_loaders


def cal_acc(loader, netF, netC):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            # outputs = netC(netB(netF(inputs)))

            outputs = netC(netF(inputs))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = (labels+1).float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, (labels+1).float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()
    return accuracy*100, mean_ent


def train_source(args):

    dset_loaders = digit_load(args)
    ## set base network
    if args.dset == 'u2m':
        netF = network.LeNetBase().cuda()
    elif args.dset == 'm2u':
        netF = network.LeNetBase().cuda()  
    elif args.dset == 's2m':
        netF = network.DTNBase().cuda()
    elif args.dset == 'seed':
        netF = network.SeedBase().cuda()


    # netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    # netB = network.feat_bootleneck(type=args.classifier, feature_dim=128, bottleneck_dim=args.bottleneck).cuda()
    # a = args.bottleneck
    # print("a###", a)
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()
    netD = network.feat_domain( domain_num = args.domain_num, bottleneck_dim= args.bottleneck, type=args.layer).cuda()
    param_group = []
    learning_rate = args.lr
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    # for k, v in netB.named_parameters():
    #     param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netD.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    acc_init = 0
    # print(dset_loaders["source_tr"].shape)
    max_iter = args.max_epoch * len(dset_loaders["source_tr"])
    interval_iter = max_iter // 10
    iter_num = 0

    netF.train()
    netC.train()
    netD.train()

    while iter_num < max_iter:
        try:
            inputs_source, labels_source = iter_source.next()
            inputs_source_domain, labels_source_domain = iter_source_domain.next()
        except:
            iter_source = iter(dset_loaders["source_tr"])
            iter_source_domain = iter(dset_loaders["source_domain"])
            inputs_source_domain, labels_source_domain = iter_source_domain.next()
            inputs_source, labels_source = iter_source.next()

        if inputs_source.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        #targets = targets.to(self.device,dtype=torch.int64)
        labels_source = (labels_source+1).long()
        inputs_source, labels_source = inputs_source.cuda(), (labels_source).cuda()
        # print(labels_source)
        inputs_source_domain, labels_source_domain = inputs_source_domain.cuda(), labels_source_domain.cuda()
        # outputs_source = netC(netB(netF(inputs_source)))
        # ouputs_domain = netD(netB(netF(inputs_source_domain)), 0.1)

        outputs_source = netC(netF(inputs_source))
        ouputs_domain = netD(netF(inputs_source_domain), 1)

        classifier_loss = loss.CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(outputs_source, labels_source)
        DMI_loss2 = DMI_loss(outputs_source, labels_source)  ##  这个要不要

        domian_loss = torch.nn.CrossEntropyLoss()
        labels_source_domain = labels_source_domain.long()
        # print(ouputs_domain.shape)
        # print(labels_source_domain.shape)
        domain_loss = domian_loss(ouputs_domain, labels_source_domain.squeeze() )  ## outputs 是否需要时三维的结果？

        # loss_total = classifier_loss* 0.1 + domain_loss * 0.9 + DMI_loss2 * 0.0003
        loss_total = classifier_loss* 0.1 + domain_loss + DMI_loss2 * 0.0003

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netC.eval()
            netD.eval()
            acc_s_tr, _ = cal_acc(dset_loaders['source_tr'], netF,  netC)
            acc_s_te, _ = cal_acc(dset_loaders['target'], netF,  netC)
            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%/ {:.2f}%; loss = {:.2f}%'.format(args.dset, iter_num, max_iter, acc_s_tr, acc_s_te, domain_loss)
            print("**************************************************")

            # args.out_file.write(log_str + '\n')
            # args.out_file.flush()
            print(log_str + '\n')


            if acc_s_te >= acc_init:
                acc_init = acc_s_te
                best_netF = netF.state_dict()
                best_netC = netC.state_dict()

    torch.save(best_netF, osp.join(args.output_dir, "source_F.pt"))
    torch.save(best_netC, osp.join(args.output_dir, "source_C.pt"))

    return netF,  netC



def train_domain(args):

    dset_loaders = digit_load(args)
    ## set base network
    if args.dset == 'u2m':
        netF = network.LeNetBase().cuda()
    elif args.dset == 'm2u':
        netF = network.LeNetBase().cuda()
    elif args.dset == 's2m':
        netF = network.DTNBase().cuda()
    elif args.dset == 'seed':
        netF = network.SeedBase().cuda()

    # netB = network.feat_bootleneck(type=args.classifier, feature_dim=128, bottleneck_dim=args.bottleneck).cuda()
    # a = args.bottleneck
    # print("a###", a)
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()
    netD = network.feat_classifier2(domain_num=args.domain_num, type=args.layer, bottleneck_dim=args.bottleneck).cuda()
    param_group = []
    learning_rate = args.lr
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    # for k, v in netB.named_parameters():
    #     param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netD.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    acc_init = 0
    # print(dset_loaders["source_tr"].shape)
    max_iter = args.max_epoch * len(dset_loaders["source_tr"])
    interval_iter = max_iter // 10
    iter_num = 0

    netF.train()
    netC.train()
    netD.train()

    while iter_num < max_iter:
        try:
            inputs_source, labels_source = iter_source.next()
            inputs_source_domain, labels_source_domain = iter_source_domain.next()
        except:
            iter_source = iter(dset_loaders["source_tr"])
            iter_source_domain = iter(dset_loaders["source_domain"])
            inputs_source_domain, labels_source_domain = iter_source_domain.next()
            inputs_source, labels_source = iter_source.next()

        if inputs_source.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        # targets = targets.to(self.device,dtype=torch.int64)
        labels_source = (labels_source + 1).long()
        inputs_source, labels_source = inputs_source.cuda(), (labels_source).cuda()
        # print(labels_source)
        inputs_source_domain, labels_source_domain = inputs_source_domain.cuda(), labels_source_domain.cuda()
        # outputs_source = netC(netB(netF(inputs_source)))
        # ouputs_domain = netD(netB(netF(inputs_source_domain)), 0.1)

        outputs_source = netC(netF(inputs_source))
        ouputs_domain = netD(netF(inputs_source_domain))

        classifier_loss = loss.CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(outputs_source,
                                                                                                        labels_source)
        DMI_loss2 = DMI_loss(outputs_source, labels_source)

        domian_loss = torch.nn.CrossEntropyLoss()
        labels_source_domain = labels_source_domain.long()
        # print(ouputs_domain.shape)
        # print(labels_source_domain.shape)
        domain_loss = domian_loss(ouputs_domain, labels_source_domain.squeeze())  ## outputs 是否需要时三维的结果？

        # loss_total = classifier_loss* 0.1 + domain_loss * 0.9 + DMI_loss2 * 0.0003
        loss_total = classifier_loss + domain_loss*0.1
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netC.eval()
            netD.eval()
            acc_s_tr, _ = cal_acc(dset_loaders['source_tr'], netF, netC)
            acc_s_d, _ = cal_acc2(dset_loaders['domain_te'], netF, netD)
            acc_s_te, _ = cal_acc(dset_loaders['target'], netF, netC)
            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%/ {:.2f}%/ {:.2f}%; loss = {:.2f}%'.format(args.dset, iter_num,
                                                                                                 max_iter, acc_s_tr, acc_s_d,
                                                                                                 acc_s_te, domain_loss)
            print("**************************************************")

            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')

            if acc_s_te >= acc_init:
                acc_init = acc_s_te
                best_netF = netF.state_dict()
                best_netC = netC.state_dict()

    torch.save(best_netF, osp.join(args.output_dir, "source_Fs.pt"))
    torch.save(best_netC, osp.join(args.output_dir, "source_Cs.pt"))




# def test_target(args):
#     dset_loaders = digit_load(args)
#     ## set base network
#     if args.dset == 'u2m':
#         netF = network.LeNetBase().cuda()
#     elif args.dset == 'm2u':
#         netF = network.LeNetBase().cuda()
#     elif args.dset == 's2m':
#         netF = network.DTNBase().cuda()
#     elif args.dset =='seed':
#         netF =network.SeedBase().cuda()
#
#
#     netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()
#     args.modelpath = args.output_dir + '/source_F_0.3.pt'
#     netF.load_state_dict(torch.load(args.modelpath))
#     # args.modelpath = args.output_dir + '/source_Fs.pt'
#     # netF2.load_state_dict(torch.load(args.modelpath))
#     args.modelpath = args.output_dir + '/source_C_0.3.pt'
#     netC.load_state_dict(torch.load(args.modelpath))
#     netF.eval()
#     netC.eval()
#
#     acc, _ = cal_acc(dset_loaders['test'], netF, netC)
#     log_str = '#####################################################fignal_Task: {}, Accuracy = {:.2f}%'.format(args.dset, acc)
#     args.out_file.write(log_str + '\n')
#     args.out_file.flush()
#     print(log_str+'\n')
#



def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def train_target(args):
    dset_loaders = digit_load(args)
    ## set base network
    if args.dset == 'u2m':
        netF = network.LeNetBase().cuda()
    elif args.dset == 'm2u':
        netF = network.LeNetBase().cuda()  
    elif args.dset == 's2m':
        netF = network.DTNBase().cuda()
    elif args.dset =='seed':
        netF = network.SeedBase().cuda()

    netF2 = network.SeedBase().cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()
    netC2 = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    args.modelpath = args.output_dir + '/source_F.pt'   
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/source_C.pt'
    netC.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/source_Fs.pt'
    netF2.load_state_dict(torch.load(args.modelpath))  ## 在这个地方更新了两个model的参数
    args.modelpath = args.output_dir + '/source_Cs.pt'
    netC2.load_state_dict(torch.load(args.modelpath))
    netC.eval()
    for k, v in netC.named_parameters():
        v.requires_grad = False
    for k, v in netC2.named_parameters():
        v.requires_grad = False

    param_group = []
    for k, v in netF2.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]  ## 是否要更新第二个model

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = len(dset_loaders["target"])
    # interval_iter = max_iter // args.interval
    iter_num = 0

    while iter_num < max_iter:
        optimizer.zero_grad()
        try:
            inputs_test, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, tar_idx = iter_test.next()  ##  every time has 30

        if inputs_test.size(0) == 1:
            continue

        if iter_num % interval_iter == 0 and args.cls_par > 0:  ##  每一轮计算一次聚类的标签
            netF.eval()
            netF2.eval()

            mem_label = obtain_label(dset_loaders['target'], netF, netF2, netC, netC2, args)
            mem_label = torch.from_numpy(mem_label).cuda()
            netF.train()
            netF2.train()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        inputs_test = inputs_test.cuda()
        # features_test = netB(netF(inputs_test))
        features_test = netF2(inputs_test)  ################# 这个地方需不要虚
        outputs_test = netC2(features_test)
        n = 24
        # print("iter_num", iter_num)
        if args.cls_par > 0:
            if iter_num % n == 0:
                pred = mem_label[(n-1)*128:3000]
            else:
                pred = mem_label[((iter_num-1) % n)*128: (iter_num % n)*128]
        #     print(iter_num)
        #     pred = mem_label[((iter_num-1)%35)*84: (iter_num%35) *84]
        #     print("pred", pred.size())
        #     print(outputs_test.size())

            classifier_loss = args.cls_par * nn.CrossEntropyLoss()(outputs_test, pred)
        else:
            classifier_loss = torch.tensor(0.0).cuda()

        if args.ent:  ## true
            # print(args.ent)
            softmax_out = nn.Softmax(dim=1)(outputs_test)
            entropy_loss = torch.mean(loss.Entropy(softmax_out))  ##  average entropys
            if args.gent:
                # print(args.gent)##True

                msoftmax = softmax_out.mean(dim=0)  ## twice mean
                entropy_loss -= torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))  ##

            im_loss = entropy_loss * args.ent_par
            classifier_loss += im_loss
            # classifier_loss += im_loss + DMI_loss(outputs_test, pred)*0.01

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netC.eval()
            netF2.eval()
            netC2.eval()
            # acc1, _ = cal_acc(dset_loaders['test'], netF, netC)
            # log_str1 = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.dset, iter_num, max_iter, acc1)
            acc2, _ = cal_acc(dset_loaders['test'], netF2, netC2)
            log_str2 = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.dset, iter_num, max_iter, acc2)
            args.out_file.write(log_str2 + '\n')
            args.out_file.flush()
            # print(log_str1+'\n')
            print(log_str2 + '\n')


    # if args.issave:
    #     torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F_" + args.savename + ".pt"))
    #     torch.save(netF2.state_dict(), osp.join(args.output_dir, "target_F2_" + args.savename + ".pt"))
    #     torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C_" + args.savename + ".pt"))

    return netF, netF2, netC, netC2

def obtain_label(loader, netF, netF2, netC, netC2, args, c=None):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            # feas = netB(netF(inputs))
            feas1 = netF(inputs)
            feas2 = netF2(inputs)
            # feas = feas1 + feas2
            feas = torch.cat((feas1, feas2), dim=1)
            # print("feas.shape", feas.shape)
            # feas = feas2
            outputs = netC2(feas2)  ##  决定了聚类的初始化点
            # outputs2 = netC2(feas2)

            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = (labels+1).float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, (labels+1).float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    # print("all_output.size()", all_output.size())
    _, or_predict = torch.max(all_output, 1)


    accuracy = torch.sum(torch.squeeze(or_predict).float() == all_label).item() / float(all_label.size()[0])
    
    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1) ## batch-size, 65
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()  ## 65, batch-size,/  return 欧几里得2范数
    all_fea = all_fea.float().cpu().numpy()

    # K = all_output.size(10) ##
    K = all_output.size(-1)  ## how k = 3
    print("index", args.index)
    aff = all_output.float().cpu().numpy()  ## batch_size * 3
    initc = aff.transpose().dot(all_fea)  ##    3* 65
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])  # 正则化， 得出每个类的向量
    dd = cdist(all_fea, initc, 'cosine') ## comput the distance of all_fea and initc  ## 求每个all 和 三个种类的距离
    pred_label = dd.argmin(axis=1)  ## Return indices of the minimum values along the given  ## 返回距离最近的下标值
    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)

    for round in range(5):
        aff = np.eye(K)[pred_label]  ## new label
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc, 'cosine')
        pred_label = dd.argmin(axis=1)
        acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)

    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy*100, acc*100)
    print("********" + log_str + '\n')
    # args.out_file.write(log_str + '\n')
    # args.out_file.flush()
    # print(log_str+'\n')
    return pred_label.astype('int')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='1', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=30, help="maximum epoch")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='seed', choices=['u2m', 'm2u','s2m'])
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--cls_par', type=float, default=0.3)   ## cluster loss
    parser.add_argument('--ent_par', type=float, default=1)   ## MIloss
    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--bottleneck', type=int, default=64)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--smooth', type=float, default=0)
    parser.add_argument('--output', type=str, default='./seed/')
    parser.add_argument('--issave', type=bool, default=True)
    parser.add_argument('--domain_num', type = int, default = 14)
    parser.add_argument('--index', type = int, help="index")
    args = parser.parse_args()
    args.class_num = 3

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    args.output_dir = osp.join(args.output, 'seed' + str(args.index))
    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # if not osp.exists(osp.join(args.output_dir + '/source_F.pt')):
    args.out_file = open(osp.join(args.output_dir, 'log_src.txt'), 'w')
    args.out_file.write(print_args(args)+'\n')
    args.out_file.flush()
    # train_source(args)
    train_domain(args)
    print("index", args.index +1)
    args.savename = 'par_' + str(args.cls_par)
    args.out_file = open(osp.join(args.output_dir, 'log_tar_' + args.savename + '.txt'), 'w')
    args.out_file.write(print_args(args)+'\n')
    args.out_file.flush()
    train_target(args)
    # test_target(args)