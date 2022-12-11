# AI programming course final project based on TransFG


Fine-grained-classification on CUB_200_2011 based on the paper:  [*TransFG: A Transformer Architecture for Fine-grained Recognition (AAAI2022)*](https://arxiv.org/abs/2103.07976)  

We used the official implementation as a backbone, while changed part selection module and delete    the Contrastive loss to improve the preformance of the model in our case(resized to 224 instead of 448 due to limited computing resource).

Furthermore, we conduct several experiments on different hyperparameter settings, and try to visualize where naive CNN pay "attention" on when classiying a image.

For more details, please check 作业文档.md. ***Feel free to contact us through [email](2000016625@stu.pku.edu.cn) if you have any questions!***


## Dependencies:
+ Python 3.7.3
+ PyTorch 1.5.1
+ torchvision 0.6.1
+ ml_collections
+ apex
+ Linux only!

## Usage
### 1. Download Google pre-trained ViT models

* [Get pre-trained ViT models in this link](https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz)

### 2. Prepare data

we only tried CUB-200-2011 in our project data which could be downloaded through this link:

+ [CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/)

Please check the position of **dataset/pretrained model**, and modify corresponding code in train.py；Experiments/segmentation.py; Experiment/CNN_attention.py

### 3. Install required packages

Install dependencies with the following command:

```bash
pip3 install -r requirements.txt
git clone https://github.com/NVIDIA/apex
cd apex
python setup.py install
```

### 4. Train

To train our model on CUB-200-2011 dataset with 1 gpus in FP-16 mode and using mul as fusion operator for 10000 steps run:

```bash
python3 -m torch.distributed.launch --nproc_per_node=1 train.py --dataset CUB_200_2011 --split overlap --num_steps 10000 --fp16 --fusion_way mul --learning_rate 1e-2 --name MULfusion
```

### 5. Experiemnt

To visualize attention of naive CNN, firstly, you need to run Experiments/Segmentation.py first:

```bash
python3  Experiments/Segmentation.py 
```

Then, you could run Experiment/CNN_attention.ipynb to train a Resnet18/50, and visualize its attention; examples are displayed in the file as well.  
