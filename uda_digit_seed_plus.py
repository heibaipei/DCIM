import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
import pickle
from DANN_all_data import load_data_subject
from torch.nn import functional as F 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

def Entropy(input_):
    bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, size_average=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.size_average = size_average
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.to(device)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        if self.size_average:
            loss = (- targets * log_probs).mean(0).sum()
        else:
            loss = (- targets * log_probs).sum(1)
        return loss


class MyDataset():
    def __init__(self, data, labels):
        self.eeg = data
        self.labels = labels

    def __getitem__(self, index):
        img, target = self.eeg[index], self.labels[index]
        return img, target, index

    def __len__(self):
        return len(self.eeg)
    

def digit_load(args):
    index = args.index
    train_bs = 128
    source_data, source_label, target_data, target_label, domain_label = load_data_subject(index)

    print("source_data", source_data.shape)
    print("source_label", source_label.shape)
    print("domain_label", domain_label.shape)
    print("target_data", target_data.shape)
    print("target_label", target_label.shape)

    train_source = MyDataset(source_data, (source_label+1))
    domian_source = MyDataset(source_data, domain_label)

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
            labels = data[1]+1
            inputs = inputs.to(device)
            outputs = netC(netF(inputs))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()
    return accuracy*100, mean_ent

def train_AUE(args):
    dset_loaders = digit_load(args)
    ## set base network
    netF = network.SeedBase().to(device)
    netF2 = network.SeedBase().to(device)


    # netB = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netR = network.recover_seed(x_dim=310, bottleneck_dim=2*args.bottleneck).to(device)

    args.modelpath = args.output_dir + '/source_F.pt'
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/source_Fs.pt'
    netF2.load_state_dict(torch.load(args.modelpath))

    netF.eval()
    netF2.eval()

    for k, v in netF.named_parameters():
        v.requires_grad = False
    for k, v in netF2.named_parameters():
        v.requires_grad = False

    param_group = []
    for k, v in netR.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]
    netR.train()
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // 10
    iter_num = 0

    rot_acc = 10
    while iter_num < max_iter:
        optimizer.zero_grad()
        try:
            inputs_test, _, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = iter_test.next()

        if inputs_test.size(0) == 1:
            continue

        inputs_test = inputs_test.to(device)

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

#         r_labels_target = np.random.randint(0, 4, len(inputs_test))
#         r_inputs_target = rotation.rotate_batch_with_labels(inputs_test, r_labels_target)
#         r_labels_target = torch.from_numpy(r_labels_target).cuda()

        r_inputs_target = inputs_test
       
        feas1 = netF(inputs_test)
        feas2 = netF2(inputs_test)
        
        # print(feas1.shape)
        # print(feas2.shape)
        r_outputs_target = netR(torch.cat((feas1, feas2), 1))

        rotation_loss = nn.MSELoss()(r_outputs_target, r_inputs_target.float())
        rotation_loss.backward() 

        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
#             netR.eval()
# #             acc_rot = cal_acc_rot(dset_loaders['target'], netF, netB, netR)
            log_str = 'Task: {}, Iter:{}/{}; Loss = {:.2f}%'.format(args.dset, iter_num, max_iter, rotation_loss)
            # args.out_file.write(log_str + '\n')
            # args.out_file.flush()
            print(log_str+'\n')
            # netR.train()

            if rot_acc > rotation_loss:
                rot_acc = rotation_loss
                best_netR = netR.state_dict()

#     log_str = 'Best Accuracy = {:.2f}%'.format(rot_acc)
#     args.out_file.write(log_str + '\n')
#     args.out_file.flush()
#     print(log_str+'\n')

    return best_netR, rot_acc



def cal_acc_rot(loader, netF, netB, netR):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0].cuda()
            r_labels = np.random.randint(0, 4, len(inputs))
            r_inputs = rotation.rotate_batch_with_labels(inputs, r_labels)
            r_labels = torch.from_numpy(r_labels)
            r_inputs = r_inputs.cuda()
           
            f_outputs = netB(netF(inputs))
            f_r_outputs = netB(netF(r_inputs))
            r_outputs = netR(torch.cat((f_outputs, f_r_outputs), 1))

            if start_test:
                all_output = r_outputs.float().cpu()
                all_label = r_labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, r_outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, r_labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    
    return accuracy*100

def train_target_rot(args):
    dset_loaders = digit_load(args)
    ## set base network
    netF = network.SeedBase()
    netF2 = network.SeedBase()


    # netB = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netR = network.recover_seed(type='linear', x_dim=310, bottleneck_dim=2*args.bottleneck).cuda()

    args.modelpath = args.output_dir + '/source_F.pt'
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/source_Fs.pt'
    netF2.load_state_dict(torch.load(args.modelpath))

    netF.eval()
    for k, v in netF.named_parameters():
        v.requires_grad = False
    netB.eval()
    for k, v in netB.named_parameters():
        v.requires_grad = False

    param_group = []
    for k, v in netR.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]
    netR.train()
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // 10
    iter_num = 0

    rot_acc = 0
    while iter_num < max_iter:
        optimizer.zero_grad()
        try:
            inputs_test, _, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = iter_test.next()

        if inputs_test.size(0) == 1:
            continue

        inputs_test = inputs_test.to(device)

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        r_labels_target = np.random.randint(0, 4, len(inputs_test))
        r_inputs_target = rotation.rotate_batch_with_labels(inputs_test, r_labels_target)
        r_labels_target = torch.from_numpy(r_labels_target).to(device)
        r_inputs_target = r_inputs_target.to(device)
       
        f_outputs = netF(inputs_test)
        f_r_outputs = netF2(r_inputs_target)
        r_outputs_target = netR(torch.cat((f_outputs, f_r_outputs), 1))

        rotation_loss = nn.MSELoss()(r_outputs_target, r_labels_target)
        rotation_loss.backward() 

        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netR.eval()
            acc_rot = cal_acc_rot(dset_loaders['target'], netF, netB, netR)
            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.dset, iter_num, max_iter, acc_rot)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')
            netR.train()

            if rot_acc < acc_rot:
                rot_acc = acc_rot
                best_netR = netR.state_dict()

    log_str = 'Best Accuracy = {:.2f}%'.format(rot_acc)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')

    return best_netR, rot_acc

def train_source(args):
    dset_loaders = digit_load(args)
    ## set base network
    if args.dset == 'u2m':
        netF = network.LeNetBase().cuda()
    elif args.dset == 'm2u':
        netF = network.LeNetBase().cuda()  
    elif args.dset == 's2m':
        netF = network.DTNBase().cuda()

    netB = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    param_group = []
    learning_rate = args.lr
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]   

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    acc_init = 0
    max_iter = args.max_epoch * len(dset_loaders["source_tr"])
    interval_iter = max_iter // 10
    iter_num = 0

    netF.train()
    netB.train()
    netC.train()

    while iter_num < max_iter:
        try:
            inputs_source, labels_source = iter_source.next()
        except:
            iter_source = iter(dset_loaders["source_tr"])
            inputs_source, labels_source = iter_source.next()

        if inputs_source.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
        outputs_source = netC(netB(netF(inputs_source)))
        classifier_loss = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(outputs_source, labels_source)            
        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            acc_s_tr, _ = cal_acc(dset_loaders['source_tr'], netF, netB, netC)
            acc_s_te, _ = cal_acc(dset_loaders['source_te'], netF, netB, netC)
            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%/ {:.2f}%'.format(args.dset, iter_num, max_iter, acc_s_tr, acc_s_te)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')

            if acc_s_te >= acc_init:
                acc_init = acc_s_te
                best_netF = netF.state_dict()
                best_netB = netB.state_dict()
                best_netC = netC.state_dict()

            netF.train()
            netB.train()
            netC.train()

    torch.save(best_netF, osp.join(args.output_dir, "source_F.pt"))
    torch.save(best_netB, osp.join(args.output_dir, "source_B.pt"))
    torch.save(best_netC, osp.join(args.output_dir, "source_C.pt"))

    return netF, netB, netC

def test_target(args):
    dset_loaders = digit_load(args)
    ## set base network
    if args.dset == 'u2m':
        netF = network.LeNetBase().cuda()
    elif args.dset == 'm2u':
        netF = network.LeNetBase().cuda()  
    elif args.dset == 's2m':
        netF = network.DTNBase().cuda()

    netB = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    args.modelpath = args.output_dir + '/source_F.pt'   
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/source_B.pt'   
    netB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/source_C.pt'   
    netC.load_state_dict(torch.load(args.modelpath))
    netF.eval()
    netB.eval()
    netC.eval()

    acc, _ = cal_acc(dset_loaders['test'], netF, netB, netC)
    log_str = 'Task: {}, Accuracy = {:.2f}%'.format(args.dset, acc)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def train_target(args):
    dset_loaders = digit_load(args)

    netF = network.SeedBase().to(device)
    netF2 = network.SeedBase().to(device)
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).to(device)
    netC2 = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).to(device)

    args.modelpath = args.output_dir + '/source_F.pt'   
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/source_Fs.pt'
    netF2.load_state_dict(torch.load(args.modelpath))  ## 在这个地方更新了两个model的参数
    args.modelpath = args.output_dir + '/source_C.pt'
    netC.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/source_Cs.pt'
    netC2.load_state_dict(torch.load(args.modelpath))



    if not args.ssl == 0:
        netR = network.recover_seed( x_dim=310, bottleneck_dim=2*args.bottleneck).to(device)
        netR_dict, acc_rot = train_AUE(args)
        netR.load_state_dict(netR_dict)

   
    netC.eval()
    netC2.eval()
    for k, v in netC.named_parameters():
        v.requires_grad = False
    for k, v in netC2.named_parameters():
        v.requires_grad = False

    param_group = []
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]
    for k, v in netF2.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]
    if not args.ssl == 0:
        for k, v in netR.named_parameters():
            param_group += [{'params': v, 'lr': args.lr}]
        netR.train()

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = len(dset_loaders["target"])
    iter_num = 0

    while iter_num < max_iter:
        optimizer.zero_grad()
        try:
            inputs_test,label, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, label, tar_idx = iter_test.next()

        if inputs_test.size(0) == 1:
            continue

        if iter_num % interval_iter == 0 and args.cls_par > 0:
            netF.eval()
            netF2.eval()
            # mem_label = obtain_label(dset_loaders['target_te'], netF, netF2, netC, args)
            mem_label = obtain_label(dset_loaders['target_te'], netF, netF2, netC, netC2, args, c=None)
            mem_label = torch.from_numpy(mem_label).to(device)
            netF.train()
            netF2.train()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        inputs_test = inputs_test.to(device)
        features_test = netF(inputs_test)
        outputs_test = netC(features_test)

        if args.cls_par > 0:
            pred = mem_label[tar_idx]
            classifier_loss = args.cls_par * nn.CrossEntropyLoss()(outputs_test, pred)
        else:
            classifier_loss = torch.tensor(0.0).to(device)

        if args.ent:
            softmax_out = nn.Softmax(dim=1)(outputs_test)
            entropy_loss = torch.mean(Entropy(softmax_out))
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                entropy_loss -= torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))

            im_loss = entropy_loss * args.ent_par
            classifier_loss += im_loss

        optimizer.zero_grad()
        classifier_loss.backward()

        if not args.ssl == 0:

            # r_labels_target = np.random.randint(0, 4, len(inputs_test))
            # r_inputs_target = rotation.rotate_batch_with_labels(inputs_test, r_labels_target)
            # r_labels_target = torch.from_numpy(r_labels_target).cuda()
            # r_inputs_target = r_inputs_target.cuda()
            feas1 = netF(inputs_test)
            feas2 = netF2(inputs_test)
            # feas = torch.concat(feas1, feas2)
            # output_rec = rec(feas)


            # f_outputs = netF(inputs_test)
            # f_outputs1 = f_outputs.detach()

            r_outputs_target = netR(torch.cat((feas1, feas2), 1))

            rotation_loss = args.ssl * nn.MSELoss()(r_outputs_target, inputs_test.float())   
            rotation_loss.backward()  # 这里开始backward

        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            acc, _ = cal_acc(dset_loaders['test'], netF, netC)
            log_str = 'Task: {}, Iter:{}/{}; sharring-Accuracy = {:.2f}%'.format(args.dset, iter_num, max_iter, acc)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')


            netF2.eval()
            acc, _ = cal_acc(dset_loaders['test'], netF2, netC2)
            log_str = 'Task: {}, Iter:{}/{}; special-Accuracy = {:.2f}%'.format(args.dset, iter_num, max_iter, acc)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')

            netF.train()
            netF2.train()


    # if args.issave:
    #     torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F_" + args.savename + ".pt"))
    #     torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C_" + args.savename + ".pt"))

    return None

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
            # feas = torch.cat((feas1, feas2), dim=1)
            feas = feas1+ feas2
            outputs1 = netC(feas1)
            outputs2 = netC2(feas2)
            outputs = outputs1 + outputs2
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
    print(log_str+'\n')
    save_path2 = args.output_dir + "/result1.npz"
    np.savez(save_path2, both=accuracy, cluster=acc)
    return pred_label.astype('int')


# def obtain_label(loader, netF, netB, netC, args, c=None):
#     start_test = True
#     with torch.no_grad():
#         iter_test = iter(loader)
#         for _ in range(len(loader)):
#             data = iter_test.next()
#             inputs = data[0]
#             labels = data[1]
#             inputs = inputs.cuda()
#             feas = netB(netF(inputs))
#             outputs = netC(feas)
#             if start_test:
#                 all_fea = feas.float().cpu()
#                 all_output = outputs.float().cpu()
#                 all_label = labels.float()
#                 start_test = False
#             else:
#                 all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
#                 all_output = torch.cat((all_output, outputs.float().cpu()), 0)
#                 all_label = torch.cat((all_label, labels.float()), 0)
#     all_output = nn.Softmax(dim=1)(all_output)
#     _, predict = torch.max(all_output, 1)
#     accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    
#     all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
#     all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
#     all_fea = all_fea.float().cpu().numpy()

#     K = all_output.size(1)
#     aff = all_output.float().cpu().numpy()
#     initc = aff.transpose().dot(all_fea)
#     initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
#     dd = cdist(all_fea, initc, 'cosine')
#     pred_label = dd.argmin(axis=1)
#     acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)

#     for round in range(1):
#         aff = np.eye(K)[pred_label]
#         initc = aff.transpose().dot(all_fea)
#         initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
#         dd = cdist(all_fea, initc, 'cosine')
#         pred_label = dd.argmin(axis=1)
#         acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)

#     log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy*100, acc*100)
#     args.out_file.write(log_str + '\n')
#     args.out_file.flush()
#     print(log_str+'\n')
#     return pred_label.astype('int')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT++')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--max_epoch', type=int, default=30, help="maximum epoch")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='s2m', choices=['u2m', 'm2u','s2m'])
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--cls_par', type=float, default=0.1)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--bottleneck', type=int, default=64)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--smooth', type=float, default=0.1)   
    parser.add_argument('--ssl', type=float, default=0.1) 
    parser.add_argument('--output', type=str, default='./seed/')
    parser.add_argument('--issave', type=bool, default=True)
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()
    args.class_num = 3

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    args.output_dir = osp.join(args.output, str(args.index))
    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # if not osp.exists(osp.join(args.output_dir + '/source_F.pt')):
    args.out_file = open(osp.join(args.output_dir, 'log_src.txt'), 'w')
    args.out_file.write(print_args(args)+'\n')
    args.out_file.flush()
    # train_source(args)
    # test_target(args)

    args.max_epoch = 15
    args.savename = 'par_' + str(args.cls_par)
    if args.ssl > 0:
        args.savename += ('_ssl_') + str(args.ssl)
    args.out_file = open(osp.join(args.output_dir, 'log_tar_' + args.savename + '.txt'), 'w')
    args.out_file.write(print_args(args)+'\n')
    args.out_file.flush()
    train_target(args)
    args.out_file.close()