
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import time
import io
import cv2

def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(15, 5, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # plt.imshow(np.transpose(npimg, (0, 2, 1)))
    plt.show()

def imshow_new(img,text_list=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text_list:
        header=50
        footer=350
        for  text in text_list:
            # plt.text(75, 8, text, style='italic',fontweight='bold', bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
            plt.text(footer, header, ("disimilarity:",text), style='italic',fontweight='bold', bbox={'facecolor':'white', 'alpha':0.6, 'pad':5})
            header+=150
            # plt.text(75, 180, text, style='italic',fontweight='bold', bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()    

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()

class Config():
    # training_dir = "/home/mayank_sati/Desktop/label_traffic_light/base_color/training/"
    # # training_dir = "/home/mayank_sati/Desktop/label_traffic_light/roi_label_bins/"
    # # testing_dir = "/home/mayank_sati/Desktop/label_traffic_light/base_color/testing/"
    # testing_dir = "/home/mayank_sati/Desktop/label_traffic_light/roi_label_bins/"
    # training_dir = "/home/mayank_sati/pycharm_projects/pytorch/siamese/Facial-Similarity-with-Siamese-Networks-in-Pytorch/data/only_traffic_light/training/"
    training_dir = "data/only_traffic_light/reference"
    testing_dir = "data/only_traffic_light/testing/"
    train_batch_size = 64
    train_number_epochs = 3
    train_batch_size = 64
    train_number_epochs = 2


class SiameseNetworkDataset(Dataset):
    
    def __init__(self,imageFolderDataset,transform=None,should_invert=True):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        self.should_invert = should_invert
        
    def __getitem__(self,index):
        # img0_tuple = random.choice(self.imageFolderDataset.imgs)
        img0_tuple = random.choice(self.imageFolderDataset.samples)

        img0 = Image.open(img0_tuple[0])
        img0 = img0.convert("RGB")
        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            # img1 = PIL.ImageOps.invert(img1)
        if self.transform is not None:
            img0 = self.transform(img0)
        # return img0, img1 , torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32))
        label=img0_tuple[1]
        return img0,label

    def __len__(self):
        return len(self.imageFolderDataset.imgs)




#image_resulation
res=30
#batch size
ref_batch=50

#reference datasets
folder_dataset = dset.ImageFolder(root=Config.training_dir)
# folder_dataset = dset.ImageFolder(root='/home/mayank_sati/Desktop/root')
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset, transform=transforms.Compose([transforms.Resize((res,res)), transforms.ToTensor()]),should_invert=False)
vis_dataloader = DataLoader(siamese_dataset, shuffle=True, num_workers=8, batch_size=ref_batch)
dataiter = iter(vis_dataloader)
reference_batch = next(dataiter)
imshow(torchvision.utils.make_grid(reference_batch[0]))



class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),


            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),


        )

        self.fc1 = nn.Sequential(
            nn.Linear(8*res*res, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive


net = SiameseNetwork().cuda()
# net=torch.load("model-epoch-49.pt")
net=torch.load("model-epoch_traffic_light-2.pt")
criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(),lr = 0.0005 )

counter = []
loss_history = [] 
iteration_number= 0



folder_dataset_test = dset.ImageFolder(root=Config.testing_dir)
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test, transform=transforms.Compose([transforms.Resize((res,res)), transforms.ToTensor()]),should_invert=False)

color=['black','green','red','yellow']


for loop in range(20):
    test_dataloader = DataLoader(siamese_dataset,num_workers=6,batch_size=1,shuffle=True)
    dataiter = iter(test_dataloader)
    x0,_ = next(dataiter)
    #############################
    input_batch = torch.repeat_interleave(x0, ref_batch, dim=0)
    ############################
    t1=time.time()
    output1,output2 = net(Variable(input_batch).cuda(),Variable(reference_batch[0]).cuda())
    print("actual time taken", (time.time()-t1)*1000)
    euclidean_distance = F.pairwise_distance(output1, output2)
    values_mx, indices_mx = euclidean_distance.max(0)
    values_mn, indices_mn = euclidean_distance.min(0)
    # print(values_mn,indices_mn)
    dismalirty=float(values_mn)
    light_color = color[reference_batch[1][indices_mn]]
#####################################################################33
    euclidean_distance = euclidean_distance.cpu().detach().numpy()
    reference_batch_np=reference_batch[0].cpu().detach().numpy()
    input_batch_np=input_batch.cpu().detach().numpy()
    imgs_comb = np.concatenate((reference_batch_np, input_batch_np), axis=2)
    # cv2.imshow('Main', imgs_comb)
#####################################################################
    # settings
    h, w = 30, 60  # for raster image
    nrows, ncols = 5, 10  # array of sub-plots
    # figsize = [6, 8]  # figure size, inches
    figsize = [10, 10]  # figure size, inches

    # prep (x,y) for extra plotting on selected sub-plots
    # xs = np.linspace(0, 2 * np.pi, 60)  # from 0 to 2pi
    # ys = np.abs(np.sin(xs))  # absolute of sine

    # create figure (fig), and array of axes (ax)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    # plot simple raster image on each sub-plot
    for i, axi in enumerate(ax.flat):
        # i runs from 0 to (nrows*ncols-1)
        # axi is equivalent with ax[rowid][colid]
        # img = np.random.randint(10, size=(h, w))
        img = imgs_comb[i]
        # img = input_batch[i]
        img=np.moveaxis(img, 0, -1)
        # plt.imshow(img)
        axi.imshow(img, alpha=1)
        # get indices of row/column
        # rowid = i // ncols
        # colid = i % ncols
        # write row/col indices as axes' title for identification
        axi.set_title("dis:" + str(round(euclidean_distance[i],3) ))
        axi.axis('off')
        # axi.set_title("Row:" + str(rowid) + ", Col:" + str(colid))

    plt.tight_layout(True)
    plt.show()

    #################################################################################
    ref = reference_batch[0][indices_mn].unsqueeze(0)
    inp = input_batch[indices_mn].unsqueeze(0)
    concatenated = torch.cat((inp, ref), 0)
    imshow(torchvision.utils.make_grid(concatenated), 'Color_of_light: {}'.format(light_color+ str(  round(dismalirty,3))))
    ############################################################################################