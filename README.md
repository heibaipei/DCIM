# EEG EMOTION RECOGNITION BASED ON CONTRASTIVE SELF-SUPERVISED LEARNING


environment need the latest Pytorch version, python=3.6  pytorch 1.10.2

for the seed dataset 

only-deepcluster

python uda_digital_seed.py --index=1  

with the ssl and mix 

python uda_digital_plus.py --index=1

with the GMM and mixup

python  CoWA_seed.py --index=1





## the SEED data can be downloaded: https://bcmi.sjtu.edu.cn/~seed/index.html  
## The DEAP dataset can be downlaod: http://www.eecs.qmul.ac.uk/mmv/datasets/deap   
## the code refer to https://github.com/tim-learn/SHOT SHOT++ and Cowv 
##　and the paper refer to [**Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation**](https://arxiv.org/abs/2002.08546)
