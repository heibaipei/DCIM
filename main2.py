
import os

# for i in range(15):
#     b = i
#     python_content = "python /home/ydwang/wangDataDisk/FADA-Pytorch-master/main2.py 5 150 50 %s %s 90 64" %(b, a)
#         #"python uda_digit.py --index=s% --cls_par=%s" %(b, a)
#     over = os.system(python_content)
#     print("over")
for a in range(15):
    over = os.system("python uda_digit.py --index=%s --cls_par=%s" %(a, 0.05))