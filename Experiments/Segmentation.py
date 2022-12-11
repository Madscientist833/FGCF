import os
import shutil
root='/home/featurize/data/'
imgdict={}
with open(root+'CUB_200_2011/images.txt','r',encoding='utf-8') as f:
    for line in f:
        split_line = line.split()
        imgdict[split_line[1]]=split_line[0]
trainsplit={}
with open(root+'CUB_200_2011/train_test_split.txt','r',encoding='utf-8') as f:
    for line in f:
        split_line = line.split()
        trainsplit[split_line[0]]=split_line[1]
print("move begin")
for images in imgdict:
    pp=images.split('/')[0]
    if trainsplit[imgdict[images]]=='1':
        if not os.path.exists(root+'CUB_200_2011/train/'+pp):
            os.makedirs(root+'CUB_200_2011/train/'+pp)
        shutil.copy(root+"CUB_200_2011/images/"+images,root+'CUB_200_2011/train/'+pp)
    else:
        if not os.path.exists(root+'CUB_200_2011/test/'+pp):
            os.makedirs(root+'CUB_200_2011/test/'+pp)
        shutil.copy(root+"CUB_200_2011/images/"+images,root+'CUB_200_2011/test/'+pp)