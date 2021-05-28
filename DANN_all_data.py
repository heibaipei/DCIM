
import numpy as np
import random
import matplotlib.pyplot as plt
# import tensorflow as tf
from mpl_toolkits.axes_grid1 import ImageGrid
from keras.utils import to_categorical, np_utils
from sklearn import preprocessing
import scipy.io as sio
from numpy import random as nr

def load_data_subject(subjectIndex):
    log_dir = "./data/data"
    s1 = np.load(log_dir + '/1.npz')  ## have separated the source and target  the 2.npz and 41.npz come from same person
    s2 = np.load(log_dir + '/2.npz')
    s3 = np.load(log_dir + '/3.npz')
    s4 = np.load(log_dir + '/4.npz')
    s5 = np.load(log_dir + '/42.npz')
    s6 = np.load(log_dir + '/6.npz')
    s7 = np.load(log_dir + '/7.npz')
    s8 = np.load(log_dir + '/8.npz')
    s9 = np.load(log_dir + '/9.npz')
    s10 = np.load(log_dir + '/42.npz')
    s11 = np.load(log_dir + '/13.npz')
    s12 = np.load(log_dir + '/24.npz')
    s13 = np.load(log_dir + '/26.npz')
    s14 = np.load(log_dir + '/31.npz')
    s15 = np.load(log_dir + '/44.npz')
    # print(s1.keys())
    # ['train_data', 'train_label', 'test_data', 'test_label']

    s1_data = np.vstack((s1["train_data"], s1["test_data"])).reshape(-1, 310)  ## first 9 person, and 6 preson
    s2_data = np.vstack((s2["train_data"], s2["test_data"])).reshape(-1, 310)
    s3_data = np.vstack((s3["train_data"], s3["test_data"])).reshape(-1, 310)
    s4_data = np.vstack((s4["train_data"], s4["test_data"])).reshape(-1, 310)
    s6_data = np.vstack((s6["train_data"], s6["test_data"])).reshape(-1, 310)
    s7_data = np.vstack((s7["train_data"], s7["test_data"])).reshape(-1, 310)
    s8_data = np.vstack((s8["train_data"], s8["test_data"])).reshape(-1, 310)
    s5_data = np.vstack((s5["train_data"], s5["test_data"])).reshape(-1, 310)
    s9_data = np.vstack((s9["train_data"], s9["test_data"])).reshape(-1, 310)
    s10_data = np.vstack((s10["train_data"], s10["test_data"])).reshape(-1, 310)
    s11_data = np.vstack((s11["train_data"], s11["test_data"])).reshape(-1, 310)
    s12_data = np.vstack((s12["train_data"], s12["test_data"])).reshape(-1, 310)
    s13_data = np.vstack((s13["train_data"], s13["test_data"])).reshape(-1, 310)
    s14_data = np.vstack((s14["train_data"], s14["test_data"])).reshape(-1, 310)
    s15_data = np.vstack((s15["train_data"], s15["test_data"])).reshape(-1, 310)

    print(s1["train_label"])
    s1_label = np.hstack((s1["train_label"], s1["test_label"]))  ## first 9 person, and 6 preson
    s2_label = np.hstack((s2["train_label"], s2["test_label"]))
    s3_label = np.hstack((s3["train_label"], s3["test_label"]))
    s4_label = np.hstack((s4["train_label"], s4["test_label"]))
    s6_label = np.hstack((s6["train_label"], s6["test_label"]))
    s7_label = np.hstack((s7["train_label"], s7["test_label"]))
    s8_label = np.hstack((s8["train_label"], s8["test_label"]))
    s5_label = np.hstack((s5["train_label"], s5["test_label"]))
    s9_label = np.hstack((s9["train_label"], s9["test_label"]))
    s10_label = np.hstack((s10["train_label"], s10["test_label"]))
    s11_label = np.hstack((s11["train_label"], s11["test_label"]))
    s12_label = np.hstack((s12["train_label"], s12["test_label"]))
    s13_label = np.hstack((s13["train_label"], s13["test_label"]))
    s14_label = np.hstack((s14["train_label"], s14["test_label"]))
    s15_label = np.hstack((s15["train_label"], s15["test_label"]))

    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    s1_data = min_max_scaler.fit_transform(s1_data[0:3394])
    s2_data = min_max_scaler.fit_transform(s2_data[0:3394])
    s3_data = min_max_scaler.fit_transform(s3_data[0:3394])
    s4_data = min_max_scaler.fit_transform(s4_data[0:3394])
    s5_data = min_max_scaler.fit_transform(s5_data[0:3394])
    s6_data = min_max_scaler.fit_transform(s6_data[0:3394])
    s7_data = min_max_scaler.fit_transform(s7_data[0:3394])
    s8_data = min_max_scaler.fit_transform(s8_data[0:3394])
    s9_data = min_max_scaler.fit_transform(s9_data[0:3394])
    s10_data = min_max_scaler.fit_transform(s10_data[0:3394])
    s11_data = min_max_scaler.fit_transform(s11_data[0:3394])
    s12_data = min_max_scaler.fit_transform(s12_data[0:3394])
    s13_data = min_max_scaler.fit_transform(s13_data[0:3394])
    s14_data = min_max_scaler.fit_transform(s14_data[0:3394])
    s15_data = min_max_scaler.fit_transform(s15_data[0:3394])
    allData = np.concatenate((s1_data, s2_data, s3_data, s4_data, s5_data, s6_data, s7_data, s8_data, s9_data, s10_data,
                              s11_data, s12_data, s13_data, s14_data, s15_data), 0)
    target = allData[subjectIndex * 3394:(subjectIndex + 1) * 3394]
    source = np.delete(allData, range(subjectIndex * 3394, (subjectIndex + 1) * 3394), 0)

    # label = sio.loadmat(log_dir + '/labels.mat')
    #     # print(label.keys())
    #     # label = label['label'] + 1
    #     # label = np_utils.to_categorical(label,3)

    all_label = np.concatenate((s1_label, s2_label, s3_label, s4_label, s5_label, s6_label, s7_label, s8_label, s9_label,
                               s10_label, s11_label, s12_label, s13_label, s14_label, s15_label), 0)
    print(all_label.shape)  ##  (50910,)
    target_label = all_label[subjectIndex * 3394:(subjectIndex + 1) * 3394]
    # target_label = np_utils.to_categorical(target_label, 3)
    source_label = np.delete(all_label, range(subjectIndex * 3394, (subjectIndex + 1) * 3394), 0)
    domain_all = []
    domain_one = np.ones((3394, 1))
    for i in range(14):
        a = domain_one * i
        domain_all.append(a)
    domain_all = np.array(domain_all)
    print(domain_all.shape)  ## 14* 3394* 1
    domain_all = np.reshape(domain_all, (-1, 1))
    print(domain_all.shape)

    index = [i for i in range(len(target))]
    random.shuffle(index)
    target = target[index]
    target_label = target_label[index]

    index = [i for i in range(len(source))]
    random.shuffle(index)
    source = source[index]
    source_label = source_label[index]
    domain_label = domain_all[index]
    # domain_label = np_utils.to_categorical(domain_label, 14)

    # source = source[0:5000]
    # source_label = source_label[0:5000]
    # source_label = np_utils.to_categorical(source_label, 3)
    '''
    a = nr.randint(0,13) #randomly select a subject as source
    source = source[a*3394:(a+1)*3394]

    index = [i for i in range(len(source))]
    random.shuffle(index)
    source = source[index]
    source_label = label[index]

    index = [i for i in range(len(target))]
    random.shuffle(index)
    target = target[index]
    target_label = label[index]
    '''
    return source, source_label, target, target_label, domain_label

