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
from DANN_all_data import load_data_subject
from loss import CLIPContrastive
import numpy as np
import torch
from scipy import signal
import math
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy.stats import norm
from sklearn.metrics import confusion_matrix
import pickle
import matplotlib
import matplotlib.pyplot as plt
import time

matplotlib.use('Agg')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")





import random
class Transform:
    def __init__(self):       
         pass


    def scaled(self, signal, factor_list):
        """"
        scale the signal
        """
        factor = round(np.random.uniform(factor_list[0],factor_list[1]),2)
        signal[0] = 1 / (1 + np.exp(-signal[0]))
        # print(signal.max())
        return signal

    def negate(self,signal):
        """
        negate the signal
        """
        signal[0] = signal[0] * (-1)
        return signal

    def hor_filp(self,signal):
        """
        flipped horizontally
        """
        hor_flipped = np.flip(signal,axis=1)
        return hor_flipped



    def cutout_resize(self,signal,pieces):
        """
                signal: numpy array (batch x window)
                pieces: number of segments along time
                cutout 1 piece
                """
        signal = signal.T
        pieces = int(np.ceil(np.shape(signal)[0] / (np.shape(signal)[0] // pieces)).tolist())  # 向上取整
        piece_length = int(np.shape(signal)[0] // pieces)
        import random
        sequence = []

        cutout = random.randint(0, pieces)
        # print(cutout)
        # sequence1 = list(range(0, cutout))
        # sequence2 = list(range(int(cutout + 1), pieces))
        # sequence = np.hstack((sequence1, sequence2))
        for i in range(pieces):
            if i == cutout:
                pass
            else:
                sequence.append(i)
        # print(sequence)

        cutout_signal = np.reshape(signal[:(np.shape(signal)[0] // pieces * pieces)],
                                     (pieces, piece_length)).tolist()

        tail = signal[(np.shape(signal)[0] // pieces * pieces):]

        cutout_signal = np.asarray(cutout_signal)[sequence]

        cutout_signal = np.hstack(cutout_signal)
        cutout_signal = np.concatenate((cutout_signal, tail[:, 0]), axis=0)

        cutout_signal = cv2.resize(cutout_signal, (1, 3072), interpolation=cv2.INTER_LINEAR)
        cutout_signal = cutout_signal.T


        return cutout_signal

    def cutout_zero(self,signal,pieces):
        """
                signal: numpy array (batch x window)
                pieces: number of segments along time
                cutout 1 piece
                """
        signal = signal.T
        ones = np.ones((np.shape(signal)[0],np.shape(signal)[1]))
        # print(ones.shape)
        # assert False
        pieces = int(np.ceil(np.shape(signal)[0] / (np.shape(signal)[0] // pieces)).tolist())  # 向上取整
        piece_length = int(np.shape(signal)[0] // pieces)


        cutout = random.randint(1, pieces)
        cutout_signal = np.reshape(signal[:(np.shape(signal)[0] // pieces * pieces)],
                                     (pieces, piece_length)).tolist()
        ones_pieces = np.reshape(ones[:(np.shape(signal)[0] // pieces * pieces)],
                                   (pieces, piece_length)).tolist()
        tail = signal[(np.shape(signal)[0] // pieces * pieces):]

        cutout_signal = np.asarray(cutout_signal)
        ones_pieces = np.asarray(ones_pieces)
        for i in range(pieces):
            if i == cutout:
                ones_pieces[i]*=0

        cutout_signal = cutout_signal * ones_pieces
        cutout_signal = np.hstack(cutout_signal)
        cutout_signal = np.concatenate((cutout_signal, tail[:, 0]), axis=0)
        cutout_signal = cutout_signal[:,None]
        cutout_signal = cutout_signal.T

        return cutout_signal
    

    def move_avg(self,a,n, mode="same"):
        # a = a.T

        result = np.convolve(a[0], np.ones((n,)) / n, mode=mode)
        return result[None,:]

    def bandpass_filter(self, x, order, cutoff, fs=100):
        result = np.zeros((x.shape[0], x.shape[1]))
        w1 = 2 * cutoff[0] / int(fs)
        w2 = 2 * cutoff[1] / int(fs)
        b, a = signal.butter(order, [w1, w2], btype='bandpass')  # 配置滤波器 8 表示滤波器的阶数
        result = signal.filtfilt(b, a, x, axis=1)
        # print(result.shape)

        return result

    def lowpass_filter(self, x, order, cutoff, fs=100):
        result = np.zeros((x.shape[0], x.shape[1]))
        w1 = 2 * cutoff[0] / int(fs)
        # w2 = 2 * cutoff[1] / fs
        b, a = signal.butter(order, w1, btype='lowpass')  # 配置滤波器 8 表示滤波器的阶数
        result = signal.filtfilt(b, a, x, axis=1)
        # print(result.shape)

        return result

    def highpass_filter(self, x, order, cutoff, fs=100):
        result = np.zeros((x.shape[0], x.shape[1]))
        w1 = 2 * cutoff[0] / int(fs)
        # w2 = 2 * cutoff[1] / fs
        b, a = signal.butter(order, w1, btype='highpass')  # 配置滤波器 8 表示滤波器的阶数
        result = signal.filtfilt(b, a, x, axis=1)
        # print(result.shape)

        return result


    def time_warp(self,signal, sampling_freq, pieces, stretch_factor, squeeze_factor):
        """
        signal: numpy array (batch x window)
        sampling freq
        pieces: number of segments along time
        stretch factor
        squeeze factor
        """
        signal = signal.T

        total_time = np.shape(signal)[0] // sampling_freq
        segment_time = total_time / pieces
        sequence = list(range(0, pieces))
        stretch = np.random.choice(sequence, math.ceil(len(sequence) / 2), replace=False)
        squeeze = list(set(sequence).difference(set(stretch)))
        initialize = True
        for i in sequence:
            orig_signal = signal[int(i * np.floor(segment_time * sampling_freq)):int(
                (i + 1) * np.floor(segment_time * sampling_freq))]
            orig_signal = orig_signal.reshape(np.shape(orig_signal)[0],64, 1)
            if i in stretch:
                output_shape = int(np.ceil(np.shape(orig_signal)[0] * stretch_factor))
                new_signal = cv2.resize(orig_signal, (1, output_shape), interpolation=cv2.INTER_LINEAR)
                if initialize == True:
                    time_warped = new_signal
                    initialize = False
                else:
                    time_warped = np.vstack((time_warped, new_signal))
            elif i in squeeze:
                output_shape = int(np.ceil(np.shape(orig_signal)[0] * squeeze_factor))
                new_signal = cv2.resize(orig_signal, (1, output_shape), interpolation=cv2.INTER_LINEAR)
                if initialize == True:
                    time_warped = new_signal
                    initialize = False
                else:
                    time_warped = np.vstack((time_warped, new_signal))
        time_warped = cv2.resize(time_warped, (1,3072), interpolation=cv2.INTER_LINEAR)
        time_warped = time_warped.T
        return time_warped
    
    def add_noise(self, signal, noise_amount):
        """
        adding noise
        """
        signal = signal.T
        noise = (0.4 ** 0.5) * np.random.normal(1, noise_amount, np.shape(signal)[0])
        noise = noise[:,None]
        noised_signal = signal + noise
        noised_signal = noised_signal.T
        # print(noised_signal.shape)
        return noised_signal
    
    
    
    def add_noise_with_SNR(self, signal, noise_amount):
        """
        adding noise
        created using: https://stackoverflow.com/a/53688043/10700812
        """
        noised_signal_R = np.zeros(signal.shape)
        target_snr_db = noise_amount  # 20
        print("signal shape : {}".format(signal.shape))
        print("signal shape : {}".format(signal.shape))
        for i in range(signal.shape[0]-1):
            #print("signal shape : {}".format(signal.shape))
            #print("i : {}".format(i))
            signal_r = signal[i,:]
            #print("signal shape : {}".format(signal.shape))
            x_watts = signal_r ** 2
            #print("x_watts shape of {0} : {1}".format(i,x_watts.shape))
            sig_avg_watts = np.mean(x_watts)
            sig_avg_db = 10 * np.log10(sig_avg_watts)  # Calculate noise then convert to watts
            noise_avg_db = sig_avg_db - target_snr_db
            noise_avg_watts = 10 ** (noise_avg_db / 10)
            mean_noise = 0
            noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts),
                                           len(x_watts))
            # Generate an sample of white noise
            #print(noise_volts.shape)
            noised_signal = signal_r + noise_volts  # noise added signal

            #print("noised_signal shape : {}".format(noised_signal.shape))

            noised_signal = noised_signal[None,:]
            noised_signal_R[i,:]=noised_signal

        return noised_signal_R
    
    
    
    
    def crop_resize(self, signal, size):
        #print(signal.shape)
        
        signal = signal.T
        size = signal.shape[0] * size
        size = int(size)
        start = random.randint(0, signal.shape[0]-size)
        crop_signal = signal[start:start + size,:]
        #print(crop_signal.shape)

        crop_signal = cv2.resize(crop_signal, (64, 640), interpolation=cv2.INTER_LINEAR)
        # print(crop_signal.shape)
        crop_signal = crop_signal.T
        #print("crop_signal.shape : {}".format(crop_signal.shape))
        return crop_signal
    
    
    def permute(self,signal, pieces):
        """
        signal: numpy array (batch x window)
        pieces: number of segments along time
        """
        #print('signal shape ; {}'.format(signal.shape))
        permuted_signal_re = np.zeros(signal.shape)
        signal = signal.T
        
        pieces = int(np.ceil(np.shape(signal)[0] / (np.shape(signal)[0] // pieces)).tolist()) #向上取整
        piece_length = int(np.shape(signal)[0] // pieces)
        #print(pieces*piece_length)
        cal = pieces*piece_length
        while cal != 640:
            pieces = random.randint(5,20)
            pieces = int(np.ceil(np.shape(signal)[0] / (np.shape(signal)[0] // pieces)).tolist()) #向上取整
            piece_length = int(np.shape(signal)[0] // pieces)
            #print(pieces*piece_length)
            cal = pieces*piece_length
            
        sequence = list(range(0, pieces))
        np.random.shuffle(sequence)
        #print(signal[:(np.shape(signal)[0] // pieces * pieces)].shape)
        for i in range(signal.shape[1]):
            #print(i)
            #print('signal shape loop ; {}'.format(signal.shape))
            # 2,640
            permuted_signal = np.reshape(signal[:(np.shape(signal)[0] // pieces * pieces),i],
                                         (pieces, piece_length)).tolist()
            #print('permuted_signal : {}'.format(len(permuted_signal)))
            tail = signal[i,(np.shape(signal)[0] // pieces * pieces):]
            
            #print('tail shape  ; {}'.format(tail.shape))
            permuted_signal = np.asarray(permuted_signal)[sequence]
            permuted_signal = np.concatenate(permuted_signal, axis=0)
            #print('permuted_signal shape  ; {}'.format(permuted_signal.shape))
            permuted_signal = np.concatenate((permuted_signal,tail), axis=0)
            permuted_signal = permuted_signal[:,None]
            permuted_signal = permuted_signal.T
            
            permuted_signal = permuted_signal[None,:]
            permuted_signal_re[i,:]=permuted_signal
            
            # print(i)
            # if i == 0 :
            #     permuted_signal_re = permuted_signal
            # else:
            #     print('permuted_signal_re shape  ; {}'.format(permuted_signal_re.shape))
            #     print('permuted_signal shape  ; {}'.format(permuted_signal.shape))
            #     permuted_signal_re = np.stack((permuted_signal_re,permuted_signal))
            # #print(permuted_signal_re.shape)
        return permuted_signal_re

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(args, optimizer, iter_num, max_iter):
    decay = (1 + args.lr_gamma * iter_num / max_iter) ** (-args.lr_power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


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


    train_source = MyDataset(source_data, source_label)
    domian_source = MyDataset(source_data, domain_label)

    dset_loaders = {}
    dset_loaders["source_tr"] = DataLoader(train_source, batch_size=train_bs, shuffle=True,
        num_workers=args.worker, drop_last=False)

    dset_loaders["source_domain"] = DataLoader(domian_source, batch_size = train_bs, shuffle=True,
                                               num_workers=args.worker, drop_last=False)
    sd_m = 3000

    train_target = MyDataset(target_data, target_label)
    test_target = MyDataset(target_data, target_label)
    dset_loaders["source_te"] = DataLoader(train_target, batch_size=train_bs, shuffle=True,
        num_workers=args.worker, drop_last=True)
    dset_loaders["target"] = DataLoader(train_target, batch_size=train_bs, shuffle=True,
        num_workers=args.worker, drop_last=False)
    dset_loaders["target_te"] = DataLoader(train_target, batch_size=train_bs, shuffle=True,  ##
        num_workers=args.worker, drop_last=False)
    dset_loaders["test"] = DataLoader(test_target, batch_size=train_bs, shuffle=True,
        num_workers=args.worker, drop_last=False)
    return dset_loaders

def gmm(all_fea, pi, mu, all_output):    
    Cov = []
    dist = []
    log_probs = []
    
    for i in range(len(mu)):
        temp = all_fea - mu[i]
        predi = all_output[:,i].unsqueeze(dim=-1)
        Covi = torch.matmul(temp.t(), temp * predi.expand_as(temp)) / (predi.sum()) + args.epsilon * torch.eye(temp.shape[1]).cuda()
        try:
            chol = torch.linalg.cholesky(Covi)
        except RuntimeError:
            Covi += args.epsilon * torch.eye(temp.shape[1]).cuda() * 100
            chol = torch.linalg.cholesky(Covi)
        chol_inv = torch.inverse(chol)
        Covi_inv = torch.matmul(chol_inv.t(), chol_inv)
        logdet = torch.logdet(Covi)
        mah_dist = (torch.matmul(temp, Covi_inv) * temp).sum(dim=1)
        log_prob = -0.5*(Covi.shape[0] * np.log(2*math.pi) + logdet + mah_dist) + torch.log(pi)[i]
        Cov.append(Covi)
        log_probs.append(log_prob)
        dist.append(mah_dist)
    Cov = torch.stack(Cov, dim=0)
    dist = torch.stack(dist, dim=0).t()
    log_probs = torch.stack(log_probs, dim=0).t()
    zz = log_probs - torch.logsumexp(log_probs, dim=1, keepdim=True).expand_as(log_probs)
    gamma = torch.exp(zz)
    
    return zz, gamma

def evaluation(loader, netF1, netC1, netF2, netC2,  args, cnt):
    start_test = True
    iter_test = iter(loader)
    for _ in tqdm(range(len(loader))):
        data = iter_test.next()
        inputs = data[0]
        labels = (data[1]+1).cuda()
        inputs = inputs.cuda()
        feas1 = netF1(inputs)
        outputs1 = netC1(feas1)
        feas2 = netF2(inputs)
        outputs2 = netC2(feas2)
        feas = feas1 + feas2
        outputs = outputs2 + outputs1
        if start_test:
            all_fea = feas.float()
            all_output = outputs.float()
            all_label = (labels).float()
            start_test = False
        else:
            all_fea = torch.cat((all_fea, feas.float()), 0)
            all_output = torch.cat((all_output, outputs.float()), 0)
            all_label = torch.cat((all_label, (labels).float()), 0)
            
    _, predict = torch.max(all_output, 1)


    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).data.item()

    # print("Model Prediction : Accuracy = {:.2f}%", accuracy_return)
    print("mean_Entropy", mean_ent)



    # all_output_logit = all_output
    all_output = nn.Softmax(dim=1)(all_output)
    all_fea_orig = all_fea
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon2), dim=1)
    unknown_weight = 1 - ent / np.log(args.class_num) ## 代表结果的熵

    # print("unknown_weight", unknown_weight)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    
    
    if args.distance == 'cosine':
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float()
    K = all_output.shape[1]
    aff = all_output.float()
    initc = torch.matmul(aff.t(), (all_fea))
    initc = initc / (1e-8 + aff.sum(dim=0)[:,None])

    if args.pickle and (cnt==0):
        data = {
            'all_fea': all_fea,
            'all_output': all_output,
            'all_label': all_label,
            'all_fea_orig': all_fea_orig,
        }
        filename = osp.join(args.output_dir, 'data_{}'.format(args.names[args.t]) + args.prefix + '.pickle')
        with open(filename, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print('data_{}.pickle finished\n'.format(args.names[args.t]))
        
        
    ############################## Gaussian Mixture Modeling #############################

    uniform = torch.ones(len(all_fea),args.class_num)/args.class_num
    uniform = uniform.cuda()

    pi = all_output.sum(dim=0)
    mu = torch.matmul(all_output.t(), (all_fea))
    mu = mu / pi.unsqueeze(dim=-1).expand_as(mu)

    zz, gamma = gmm((all_fea), pi, mu, uniform)
    pred_label = gamma.argmax(dim=1)
    
    for round in range(1):
        pi = gamma.sum(dim=0)
        mu = torch.matmul(gamma.t(), (all_fea))
        mu = mu / pi.unsqueeze(dim=-1).expand_as(mu)

        zz, gamma = gmm((all_fea), pi, mu, gamma)
        pred_label = gamma.argmax(axis=1)
            
    aff = gamma
    
    acc = (pred_label==all_label).float().mean()
    log_str = 'Model Prediction : Accuracy = {:.2f}%'.format(accuracy * 100) + '\n'
    log_str2 = 'GMM prediction: acc = {:.2f}%'.format(acc*100) + '\n'


    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str)
    print(log_str2)
    
    ############################## Computing JMDS score #############################

    sort_zz = zz.sort(dim=1, descending=True)[0]
    zz_sub = sort_zz[:,0] - sort_zz[:,1]
    
    LPG = zz_sub / zz_sub.max()

    if args.coeff=='JMDS':
        PPL = all_output.gather(1, pred_label.unsqueeze(dim=1)).squeeze()
        JMDS = (LPG * PPL)
    elif args.coeff=='PPL':
        JMDS = all_output.gather(1, pred_label.unsqueeze(dim=1)).squeeze()
    elif args.coeff=='NO':
        JMDS=torch.ones_like(LPG)
    else:
        JMDS = LPG

    sample_weight = JMDS  
    return aff, sample_weight, accuracy
    
def KLLoss(input_, target_, coeff, args):
    softmax = nn.Softmax(dim=1)(input_)
    kl_loss = (- target_ * torch.log(softmax + args.epsilon2)).sum(dim=1)
    kl_loss *= coeff
    return kl_loss.mean(dim=0)


# def mixup(x, c_batch, t_batch, netF, netB, netC, args):
#     # weight mixup
#     if args.alpha==0:
#         outputs = netC(netB(netF(x)))
#         return KLLoss(outputs, t_batch, c_batch, args)
#     lam = (torch.from_numpy(np.random.beta(args.alpha, args.alpha, [len(x)]))).float().cuda()
#     t_batch = torch.eye(args.class_num)[t_batch.argmax(dim=1)].cuda()
#     shuffle_idx = torch.randperm(len(x))
#     mixed_x = (lam * x.permute(1,2,3,0) + (1 - lam) * x[shuffle_idx].permute(1,2,3,0)).permute(3,0,1,2)  ## 通道的混合融合
#     mixed_c = lam * c_batch + (1 - lam) * c_batch[shuffle_idx]
#     mixed_t = (lam * t_batch.permute(1,0) + (1 - lam) * t_batch[shuffle_idx].permute(1,0)).permute(1,0)
#     mixed_x, mixed_c, mixed_t = map(torch.autograd.Variable, (mixed_x, mixed_c, mixed_t))
#     mixed_outputs = netC(netB(netF(mixed_x)))
#     return KLLoss(mixed_outputs, mixed_t, mixed_c, args)


def mixup(x, c_batch, t_batch, netF, netC, args, iter_num):



    # weight mixup
    if args.alpha==0:
        outputs = netC(netF(x))
        return KLLoss(outputs, t_batch, c_batch, args)

    lam = (torch.from_numpy(np.random.beta(args.alpha, args.alpha, [len(x)]))).float().cuda()
    ## 生成概率值为beta分布的的概率

    t_batch = torch.eye(args.class_num)[t_batch.argmax(dim=1)].cuda()
    shuffle_idx = torch.randperm(len(x))


    mixed_x = (lam * x.permute(1,0) + (1 - lam) * x[shuffle_idx].permute(1,0))
    mixed_c = lam * c_batch + (1 - lam) * c_batch[shuffle_idx]
    mixed_t = (lam * t_batch.permute(1,0) + (1 - lam) * t_batch[shuffle_idx].permute(1,0))
    mixed_x, mixed_c, mixed_t = map(torch.autograd.Variable, (mixed_x.permute(1,0), mixed_c, mixed_t.permute(1,0)))
    mixed_outputs = netC(netF(mixed_x))

    return KLLoss(mixed_outputs, mixed_t, mixed_c, args)

def train_target(args):
    tenp_output_dir=args.output_dir
    netF1 = network.SeedBase().to(device)
    netF2 = network.SeedBase().to(device)
    netC1 = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).to(device)
    netC2 = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).to(device)
   
    args.modelpath = tenp_output_dir + '/source_F.pt'   
    netF1.load_state_dict(torch.load(args.modelpath))
    args.modelpath = tenp_output_dir + '/source_C.pt'   
    netC1.load_state_dict(torch.load(args.modelpath))
    args.modelpath = tenp_output_dir + '/source_Fs.pt'
    netF2.load_state_dict(torch.load(args.modelpath))  ## 在这个地方更新了两个model的参数
    args.modelpath = tenp_output_dir + '/source_Cs.pt'
    netC2.load_state_dict(torch.load(args.modelpath))

    for k, v in netC1.named_parameters():
        v.requires_grad = False
    for k, v in netC2.named_parameters():
        v.requires_grad = False

    param_group = []
    for k, v in netF2.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]
    for k, v in netF1.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]  ##     
    
    # ####################################################################
    
    # for k, v in netC1.named_parameters():
    #     if args.lr_decay3 > 0:
    #         param_group += [{'params': v, 'lr': args.lr * args.lr_decay3}]
    #     else:
    #         v.requires_grad = False
            
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)   


    dset_loaders = digit_load(args)
    cnt = 0
    
    epochs = []
    accuracies = []    
    netF1.eval()
    netC1.eval()
    netF2.eval()
    netC2.eval()
    with torch.no_grad():
        # Compute JMDS score at offline & evaluation.
        soft_pseudo_label, coeff, accuracy = evaluation(
            dset_loaders['target'], netF1,  netC1,  netF1,  netC1, args, cnt
        )
        epochs.append(cnt)
        accuracies.append(np.round(accuracy*100, 2))
    netF1.train()
    netC1.train()
    netF2.train()
    netC2.train()   

    uniform_ent = np.log(args.class_num)      
    args.max_epoch2 = 60
    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // (args.interval)
    iter_num = 0
    
    print('\nTraining start\n')
    

    while iter_num < max_iter:
        optimizer.zero_grad()
        try:
            inputs_test, label, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, label, tar_idx= iter_test.next()  ##  every time has 30

        if inputs_test.size(0) == 1:
            continue
        

        # factor = random.uniform(128)
        Trans = Transform()

        # temple = torch.reshape(inputs_test,(inputs_test.shape[0], 32, 4 ))
        # temple= temple.permute(1,0,2)
        # temple[31] = temple[31] * 0
        augment1 = Trans.add_noise(inputs_test, 10)
        # augment1 = torch.reshape(augment1, (inputs_test.shape[0],-1))      


        iter_num += 1
        lr_scheduler(args, optimizer, iter_num=iter_num, max_iter=max_iter)
        pred = soft_pseudo_label[tar_idx]
        pred_label = pred.argmax(dim=1)
        
        coeff, pred = map(torch.autograd.Variable, (coeff, pred))

        images1 = torch.autograd.Variable(augment1)
        images1 = images1.cuda()
        coeff = coeff.cuda()
        pred = pred.cuda()
        pred_label = pred_label.cuda()

        # CoWA_loss = mixup(images1, coeff[tar_idx], pred, netF, netB, netC, args)
        
        CoWA_loss1 = mixup(images1, coeff[tar_idx], pred, netF1,  netC1, args, iter_num)
        CoWA_loss2 = mixup(images1, coeff[tar_idx], pred, netF2,  netC2, args, iter_num)

        CoWA_loss = (CoWA_loss1 + CoWA_loss2) * 0.5
        # For warm up the start.
        if iter_num < args.warm * interval_iter + 1:
            CoWA_loss *= 1e-6
            
        optimizer.zero_grad()
        CoWA_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            print('Evaluation iter:{}/{} start.'.format(iter_num, max_iter))
            log_str = 'Task: {}, Iter:{}/{};'.format(args.name, iter_num, max_iter)
            # args.out_file.write(log_str + '\n')
            # args.out_file.flush()
            print(log_str)
            
            netF1.eval()
            netC1.eval()   
            netF2.eval()
            netC2.eval()         
            cnt += 1
            with torch.no_grad():
                # Compute JMDS score at offline & evaluation.
                soft_pseudo_label, coeff, accuracy = evaluation(dset_loaders["test"], netF1, netC1, netF2, netC2, args, cnt)
                epochs.append(cnt)
                accuracies.append(np.round(accuracy*100, 2))

            print('Evaluation iter:{}/{} finished.\n'.format(iter_num, max_iter))
            netF1.train()
            netC1.train()
            netF2.train()
            netC2.train()

    ####################################################################
    # if args.issave:   
    #     torch.save(netF.state_dict(), osp.join(args.output_dir, 'ckpt_F_' + args.prefix + ".pt"))
    #     torch.save(netB.state_dict(), osp.join(args.output_dir, 'ckpt_B_' + args.prefix + ".pt"))
    #     torch.save(netC.state_dict(), osp.join(args.output_dir, 'ckpt_C_' + args.prefix + ".pt"))
        
        
    log_str = '\nAccuracies history : {}\n'.format(accuracies)
    # args.out_file.write(log_str)
    # args.out_file.flush()
    print(log_str)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(epochs, accuracies, 'o-')
    args.output_dir='./'
    plt.savefig(osp.join(args.output_dir,'png_{}.png'.format(args.index)))
    plt.close()
    
    return None

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=50, help="max iterations")
    parser.add_argument('--interval', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='seed', choices=['seed', 'deap'])
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="alexnet, vgg16, resnet50, res101")
    parser.add_argument('--seed', type=int, default=2022, help="random seed")
 
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--warm', type=float, default=0.1)
    parser.add_argument('--coeff', type=str, default='LPG', choices=['LPG', 'JMDS', 'PPL','NO'])  # 'LPG' is 
    parser.add_argument('--pickle', default=False, action='store_true')
    parser.add_argument('--lr_gamma', type=float, default=20.0)
    parser.add_argument('--lr_power', type=float, default=0.75)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)
    parser.add_argument('--lr_decay3', type=float, default=0.1)

    parser.add_argument('--bottleneck', type=int, default=64)
    parser.add_argument('--epsilon', type=float, default=1e-6)
    parser.add_argument('--epsilon2', type=float, default=1e-6)
    parser.add_argument('--delta', type=float, default=2.0)
    parser.add_argument('--n_power', type=int, default=1)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])   

    parser.add_argument('--output_src', type=str, default='./seed/')
    parser.add_argument('--da', type=str, default='uda', choices=['uda'])
    parser.add_argument('--issave', type=bool, default=True)
    parser.add_argument('--output', type=str, default='./seed/')
    parser.add_argument('--index', type=int, default=1)
    args = parser.parse_args()

        
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed

    
    
    args.class_num = 3
    args.name = args.name = args.index
    ############# If you want to obtain the stochastic result, comment following lines. #############
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    # torch.cuda.manual_seed_all(SEED) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    random.seed(SEED)
    
    # for i in range(len(args.names)):
    #     start = time.time()
    #     if i == args.s:
    #         continue
    #     args.t = i

    #     folder = './data/'
    #     args.s_dset_path = folder + args.dset + '/' +args.names[args.s] + '_list.txt'
    #     args.t_dset_path = folder + args.dset + '/' + args.names[args.t] + '_list.txt'
    #     args.test_dset_path = folder + args.dset + '/' + args.names[args.t] + '_list.txt'

    args.output_dir_src = osp.join(args.output_src, str(args.index))
    # args.output_dir = osp.join(args.output, args.da, args.dset, args.names[args.s][0].upper()+args.names[args.t][0].upper())
    # args.name = args.names[args.s][0].upper()+args.names[args.t][0].upper()

    # if not osp.exists(args.output_dir):
    #     os.system('mkdir -p ' + args.output_dir)
    # if not osp.exists(args.output_dir):
    #     os.mkdir(args.output_dir)

    args.output_dir = osp.join(args.output, str(args.index))
    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)

    
    # args.prefix = '{}_alpha{}_lr{}_epoch{}_interval{}_seed{}_warm{}'.format(
    #     args.coeff, args.alpha, args.lr, args.max_epoch, args.interval, args.seed, args.warm
    # )
    
    ####################################################################
    if not osp.exists(osp.join(args.output_dir, 'ckpt_F_' + str(args.index) + ".pt")):
        args.out_file = open(osp.join(args.output_dir, 'log' + str(args.index) + '.txt'), 'w')
        args.out_file.write(print_args(args)+'\n')
        args.out_file.flush()
    train_target(args)

#     total_time = time.time() - start
#     log_str = 'Consumed time : {} h {} m {}s'.format(total_time // 3600, (total_time // 60) % 60, np.round(total_time % 60, 2))
#     # args.out_file.write(log_str + '\n')
#     # args.out_file.flush()
#     # print(log_str)
# else:
#     print('{} Already exists'.format(osp.join(args.output_dir, 'log' + args.prefix + '.txt')))
