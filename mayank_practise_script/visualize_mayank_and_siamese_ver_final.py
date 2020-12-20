import argparse
from anchors import Anchors
import sys
import cv2
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps
import torch.nn as nn
import torch.nn.functional as F
import time
import io
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
import torchvision
from dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer


# assert torch.__version__.split('.')[1] == '4'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
	res=30

	def imshow(img, text=None, should_save=False):
		npimg = img.numpy()
		plt.axis("off")
		if text:
			plt.text(15, 5, text, style='italic', fontweight='bold',
					 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
		plt.imshow(np.transpose(npimg, (1, 2, 0)))
		# plt.imshow(np.transpose(npimg, (0, 2, 1)))
		plt.show()
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
				nn.Linear(8 * res * res, 500),
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
	class SiameseNetworkDataset(Dataset):

		def __init__(self, imageFolderDataset, transform=None, should_invert=True):
			self.imageFolderDataset = imageFolderDataset
			self.transform = transform
			self.should_invert = should_invert

		def __getitem__(self, index):
			img0_tuple = random.choice(self.imageFolderDataset.imgs)
			# img0_tuple = random.choice(self.imageFolderDataset.samples)

			img0 = Image.open(img0_tuple[0])
			img0 = img0.convert("RGB")
			if self.should_invert:
				img0 = PIL.ImageOps.invert(img0)
			# img1 = PIL.ImageOps.invert(img1)
			if self.transform is not None:
				img0 = self.transform(img0)
			# return img0, img1 , torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32))
			label = img0_tuple[1]
			return img0, label

		def __len__(self):
			return len(self.imageFolderDataset.imgs)

	parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

	parser.add_argument('--dataset',default="csv", help='Dataset type, must be one of csv or coco.')
	parser.add_argument('--coco_path', help='Path to COCO directory')
	# parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
	parser.add_argument('--csv_val',default='gwm_data_all_forRetina_osl.csv', help='Path to file containing validation annotations (optional, see readme)')
	parser.add_argument('--csv_classes', default="osl.csv", help='Path to file containing class list (see readme)')
	parser.add_argument('--model',default='csv_retinanet_0.pt', help='Path to model (.pt) file.')
	# parser.add_argument('--model',default='model_final.pt', help='Path to model (.pt) file.')

	parser = parser.parse_args(args)

	if parser.dataset == 'coco':
		dataset_val = CocoDataset(parser.coco_path, set_name='val2017', transform=transforms.Compose([Normalizer(), Resizer()]))
	elif parser.dataset == 'csv':
		dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Resizer()]))
	else:
		raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

	sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
	dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)

	retinanet = torch.load(parser.model)
	use_gpu = True

	if use_gpu:
		retinanet = retinanet.cuda()

	retinanet.eval()

	unnormalize = UnNormalizer()

	###########################################33333
	#working on anchor
	anchors = Anchors()
	fake_img=torch.ones([1,3,512,512], dtype=torch.float32)#, device=cuda0)
	gen_anchor1=anchors(fake_img)
	# print(gen_anchor)
	#################################################
	anchors = Anchors()
	fake_img = torch.ones([1, 3, 480, 512], dtype=torch.float32)  # , device=cuda0)
	gen_anchor2 = anchors(fake_img)
	##############################################################

	def draw_caption(image, box, caption):
		b = np.array(box).astype(int)
		cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
		cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
	########################################################33
	# this is dataloading respect to siamese network
		# reference datasets
	# image_resolution
	res = 30
	# batch size
	ref_batch = 50

	folder_dataset = dset.ImageFolder(root='data/only_traffic_light/reference')
	# folder_dataset = dset.ImageFolder(root='/home/mayank_sati/Desktop/root')
	siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset, transform=transforms.Compose(
		[transforms.Resize((res, res)), transforms.ToTensor()]), should_invert=False)
	vis_dataloader = DataLoader(siamese_dataset, shuffle=True, num_workers=8, batch_size=ref_batch)
	dataiter = iter(vis_dataloader)
	reference_batch = next(dataiter)

	net = SiameseNetwork().cuda()
	net.load_state_dict(torch.load('model-save_dict_osl-2.pt'))
	net.eval()

	color = ['black', 'green', 'red', 'yellow']

	for idx, data in enumerate(dataloader_val):

		with torch.no_grad():
			print('********************************************************\n')
			st = time.time()
			scores, classification, transformed_anchors = retinanet(data['img'].cuda().float(),gen_anchor1,gen_anchor2)
			# print('Elapsed time: {}'.format((time.time()-st)*1000))
			print('Elapsed time detection: ',((time.time()-st)*1000))
			idxs = np.where(scores>0.01)
			img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()
			img[img<0] = 00
			img[img>255] = 255

			img = np.transpose(img, (1, 2, 0))
			img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
		#############################
		for j in range(idxs[0].shape[0]):
			bbox = transformed_anchors[idxs[0][j], :]
		# bbox=transformed_anchors.data[0,:]
			x1 = int(bbox[0])
			y1 = int(bbox[1])
			x2 = int(bbox[2])
			y2 = int(bbox[3])
			break

		if idxs[0].shape[0]==0:
			# print(0)
			continue


		# here I am adding siamese		 network
		frame = img[int(y1):int(y2), int(x1):int(x2)]
		cv2.imwrite("myimage.jpg",frame)
		# ################################33
		img0 = Image.open("myimage.jpg")
		img0 = img0.convert("RGB")
		transform = transforms.Compose([transforms.Resize((30, 30)), transforms.ToTensor()])
		img0 = transform(img0)
		input_batch = img0.repeat(ref_batch, 1, 1, 1)
		############################
		t1 = time.time()
		output1, output2 = net(Variable(input_batch).cuda(), Variable(reference_batch[0]).cuda())
		# print("actual time taken", (time.time() - t1) * 1000)
		euclidean_distance = F.pairwise_distance(output1, output2)
		values_mx, indices_mx = euclidean_distance.max(0)
		values_mn, indices_mn = euclidean_distance.min(0)
		# print(values_mn,indices_mn)
		dismalirty = float(values_mn)
		light_color = color[reference_batch[1][indices_mn]]
		print('Elapsed time with osl: ', ((time.time() - st) * 1000))
############################################################################################
		for index,j in enumerate(range(idxs[0].shape[0])):
			bbox = transformed_anchors[idxs[0][j], :]
			x1 = int(bbox[0])
			y1 = int(bbox[1])
			x2 = int(bbox[2])
			y2 = int(bbox[3])
			# label_name = dataset_val.labels[int(classification[idxs[0][j]])]
			draw_caption(img, (x1, y1, x2, y2), light_color)
			cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
			# print(label_name)
			if index>10:
				break
		# print(1)
		cv2.imshow('img', img)
		cv2.waitKey(1000)
		# cv2.waitKey(1)
		cv2.destroyAllWindows()


if __name__ == '__main__':
 main()