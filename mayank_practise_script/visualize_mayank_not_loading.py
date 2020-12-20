import numpy as np
import torchvision
import time
import os
import copy
import pdb
import time
import argparse
from anchors import Anchors
import sys
import cv2
import model
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
import torchvision.datasets as dset

from dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer


# assert torch.__version__.split('.')[1] == '4'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
	parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

	parser.add_argument('--dataset',default="csv", help='Dataset type, must be one of csv or coco.')
	parser.add_argument('--coco_path', help='Path to COCO directory')
	# parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
	parser.add_argument('--csv_val',default='gwm_data_all_forRetina_osl.csv', help='Path to file containing validation annotations (optional, see readme)')
	parser.add_argument('--csv_classes', default="osl.csv", help='Path to file containing class list (see readme)')
	parser.add_argument('--model',default='csv_retinanet_38.pt', help='Path to model (.pt) file.')
	# parser.add_argument('--model',default='model_final.pt', help='Path to model (.pt) file.')

	parser = parser.parse_args(args)

	if parser.dataset == 'coco':
		dataset_val = CocoDataset(parser.coco_path, set_name='val2017', transform=transforms.Compose([Normalizer(), Resizer()]))
	elif parser.dataset == 'csv':
		dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Resizer()]))
		dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Resizer()]))
		dataset_val=dset.ImageFolder(root='/home/mayank_sati/Desktop/root')#, transform=transforms.Compose([Normalizer(), Resizer()]))
		# dataset_val=torchvision.datasets.PhotoTour(root='/home/mayank_sati/Desktop/Baidu_TL_dataset1', name='cool', train=True, transform=transforms.Compose([Normalizer(), Resizer()]), download=False)

		#####################################3333
		# folder_dataset = dset.ImageFolder(root='/home/mayank_sati/Desktop/root')
		# siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset, transform=transforms.Compose(
		# 	[transforms.Resize((res, res)), transforms.ToTensor()]), should_invert=False)
		# vis_dataloader = DataLoader(siamese_dataset, shuffle=True, num_workers=8, batch_size=1)
		# dataiter = iter(vis_dataloader)
		# reference_batch = next(dataiter)
		from torch.utils.data import DataLoader
		loader = DataLoader(dataset_val,shuffle=True, num_workers=8, batch_size=1)
		dataiter = iter(loader)
		reference_batch = next(dataiter)
		# for dat in loader:
		# 	print(1)
		############################################

	else:
		raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

	sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
	dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)

	retinanet = torch.load(parser.model)
	# retinanet = model.resnet18(num_classes=1, pretrained=True)

	use_gpu = True

	if use_gpu:
		retinanet = retinanet.cuda()

	retinanet.eval()

	unnormalize = UnNormalizer()

	###########################################33333
	#working on anchor
	anchors = Anchors()
	fake_img=torch.ones([1,3,512,512], dtype=torch.float32)#, device=cuda0)
	gen_anchor=anchors(fake_img)
	# print(gen_anchor)
	#################################################

	def draw_caption(image, box, caption):

		b = np.array(box).astype(int)
		cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
		cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

	for idx, data in enumerate(dataloader_val):

		with torch.no_grad():
			print('********************************************************\n')
			st = time.time()

			scores, classification, transformed_anchors = retinanet(data['img'].cuda().float(),gen_anchor)
			# print('Elapsed time: {}'.format((time.time()-st)*1000))
			print('Elapsed time: ',((time.time()-st)*1000))
			idxs = np.where(scores>0.005)
			img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()

			img[img<0] = 0
			img[img>255] = 255

			img = np.transpose(img, (1, 2, 0))

			img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

			for j in range(idxs[0].shape[0]):
				bbox = transformed_anchors[idxs[0][j], :]
				x1 = int(bbox[0])
				y1 = int(bbox[1])
				x2 = int(bbox[2])
				y2 = int(bbox[3])
				label_name = dataset_val.labels[int(classification[idxs[0][j]])]
				draw_caption(img, (x1, y1, x2, y2), label_name)

				cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
				# print(label_name)

			cv2.imshow('img', img)
			cv2.waitKey(1000)
			# cv2.waitKey(1)
			cv2.destroyAllWindows()



if __name__ == '__main__':
 main()