import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import time
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Function
from PIL import Image
import numpy as np
import torchvision
import C3D_model_with_features
import random

import math
from ranksim import batchwise_ranking_regularizer
import argparse


parser = argparse.ArgumentParser("C3D_backbone")
# batchwise ranking regularizer
parser.add_argument('--regularization_weight', type=float, default=1.0, help='weight of the regularization term')
parser.add_argument('--interpolation_lambda', type=float, default=1.0, help='interpolation strength')

parser.add_argument("--dataset",default="avec2014")
parser.add_argument("--save_file",default="C3D_baseline_2014")
parser.add_argument("--feature",default= False)
parser.add_argument("--use_buffer",default= False)
parser.add_argument("--use_ranksim",default= False)
parser.add_argument("--norm",type = int ,default = 1.) 
###pre_vision norm 45
### img 高斯归一化 
args = parser.parse_args()

if os.path.exists(args.save_file):
    if len(os.listdir(args.save_file)) >3 :  
        exit(0)
else:
    os.mkdir(args.save_file)
# torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(3407)

def clip_randomcrop(img):
    w = random.choice(range(32))
    h = random.choice(range(32))
    imgs = img[:,w:224+w,h:h+224,:]
    return imgs

def clip_centercrop(img):
    imgs = img[:,16:240,16:240,:]
    return imgs

def clip_resize(img):
    imgs = np.zeros((16,112,112,3))
    for i in range(16):
        sub_img = Image.fromarray(img[i,:,:,:].astype('uint8')).convert('RGB')
        sub_img = sub_img.resize((112, 112),Image.LANCZOS)
        imgs[i,:,:,:] = sub_img
    return imgs
        
def video_to_names(meta_file):

	sub_videos = open(meta_file).readlines()

	videos = []
	video_nums = {}

	for sub_video in sub_videos:
		
		video = sub_video.split()[0].split('/')[1]

		if video not in videos:
			videos.append(video)
			video_nums[video] = 1
		else:
			video_nums[video] = video_nums[video] + 1
	
	return video , video_nums

class Estimate_Video(object):

	def __init__(self, meta_file):
		sub_videos = open(meta_file).readlines()

		self.videos = []
		self.video_nums = {}
		self.numbers = len(sub_videos)

		for sub_video in sub_videos:
			video = sub_video.split()[0].split('/')[1]

			if video not in self.videos:
				self.videos.append(video)
				self.video_nums[video] = 1
			else:
				self.video_nums[video] = self.video_nums[video] + 1
	
	def length(self):
		return self.numbers
	
	def estimate(self, predictions, labels):
		avg_predictions = []
		avg_labels = []

		start = 0
		for video in self.videos:
			avg_predictions.append(torch.mean(predictions[start:start+self.video_nums[video]]))
			avg_labels.append(torch.mean(labels[start:start+self.video_nums[video]]))
			start = start + self.video_nums[video]
		
		avg_labels = torch.Tensor(avg_labels)
		avg_predictions = torch.Tensor(avg_predictions)
		
		mae = torch.mean(torch.abs(avg_labels - avg_predictions))
		mse = torch.sqrt(torch.mean(torch.square(avg_labels - avg_predictions)))

		return mae, mse
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 自定义图片图片读取方式，可以自行增加resize、数据增强等操作    

def MyLoader(path):

    if args.dataset == "avec2014":
        outfile = Image.open("/dataset/AVEC2014/"+path)
    else:
        outfile = Image.open("/dataset/AVEC2013/AVEC2013/"+path)

    return outfile.convert('RGB')

class MyDataset (Dataset):
    # 构造函数设置默认参数
    def __init__(self, txt, transform=None, target_transform=None, loader=MyLoader):
        with open(txt, 'r') as fh:
            imgs = []
            for line in fh:
                line = line.strip('\n')  # 移除字符串首尾的换行符
                line = line.rstrip()  # 删除末尾空
                words = line.split( )  # 以空格为分隔符 将字符串分成
                imgs.append((words[0:-1], int(words[-1]))) # imgs中包含有图像路径和标签
        # print(imgs[0])
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        

    def __getitem__(self, index):
        valor = np.random.random()
        if valor < 0.25:
            flagFlip = 1
        elif valor < 0.50 and valor >= 0.25:
            flagFlip = 2
            p = random.choice(range(-15,15))
        # elif valor < 0.75 and valor >= 0.50:
        #     flagFlip = 3
        else:
            flagFlip = 4

        fn, label = self.imgs[index]
        # 调用定义的loader方法
        img = np.zeros((16,256,256,3))
        for i in range(16):
            if flagFlip == 1:
                sub_imgs = self.loader(fn[i])
                sub_imgs = sub_imgs.transpose(Image.FLIP_LEFT_RIGHT)
            elif flagFlip == 2:
                sub_imgs = self.loader(fn[i])
                sub_imgs = sub_imgs.rotate(p)            
            # elif flagFlip == 3:
            #     sub_imgs = self.loader(fn[i])
            #     sub_imgs = sub_imgs.transpose(Image.FLIP_TOP_BOTTOM)
            else:
                sub_imgs = self.loader(fn[i])
            if self.transform is not None:
                sub_imgs = self.transform(sub_imgs)
            img[i,:,:,:]= sub_imgs
        crop_img = clip_randomcrop(img)
        resize_img = clip_resize(crop_img)

        # return (resize_img - resize_img.mean())/resize_img.std() ,label/args.norm
        return resize_img/255,label/args.norm

    def __len__(self):
        return len(self.imgs)
   
class MyDataset_test (Dataset):
    # 构造函数设置默认参数
    def __init__(self, txt, transform=None, target_transform=None, loader=MyLoader):
        with open(txt, 'r') as fh:
            imgs = []
            for line in fh:
                line = line.strip('\n')  # 移除字符串首尾的换行符
                line = line.rstrip()  # 删除末尾空
                words = line.split( )  # 以空格为分隔符 将字符串分成
                imgs.append((words[0:-1], int(words[-1]))) # imgs中包含有图像路径和标签
        # print(imgs[0])
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        # 调用定义的loader方法
        img = np.zeros((16,256,256,3))
        for i in range(16):
            sub_imgs = self.loader(fn[i])
            if self.transform is not None:
                sub_imgs = self.transform(sub_imgs)
            img[i,:,:,:]= sub_imgs  
        crop_img = clip_centercrop(img)
        resize_img = clip_resize(crop_img)

        return resize_img/255,label

    def __len__(self):
        return len(self.imgs)

def train(model,train_loader,epoch,optimizer,loss_list,listMAE,listRMSE,val_gate):
    temp_gate = val_gate
    model.train()
# training-----------------------------
    train_batch_losses = []

    for step,(batch_x, batch_y) in enumerate(train_loader):
        encodings, labels = [], []
        batch_x, batch_y = Variable(batch_x).to(device), Variable(batch_y).to(device)
        batch_x=batch_x.permute(0,4,1,2,3)
        batch_x=batch_x.to(torch.float32)
        batch_y=batch_y.to(torch.float32)
        y = torch.linspace(0, 45/args.norm, 46).to(device)

        if args.feature:
            if args.use_buffer:
            # batch_y=batch_y.unsqueeze(-1)
                feature,out,center = model(batch_x,epoch, batch_y)
            else:
                feature,out = model(batch_x,epoch, batch_y)
        else:
            out = model(batch_x,epoch, batch_y)

            
        if args.use_buffer:
            loss = loss_func(out,batch_y) + loss_func(center,y)
        else:
            loss = loss_func(out,batch_y)
        

        if args.use_ranksim:    
            batch_y=batch_y.unsqueeze(-1)
            if args.use_buffer:
                if epoch > 10:
                    # batch_y=batch_y.unsqueeze(-1)
                    new_feature = torch.cat((feature,model.FDS.smoothed_mean_last_epoch),dim = 0)
                    batch_y = torch.cat((batch_y, y.unsqueeze(-1)),dim = 0)              
                    loss += (args.regularization_weight * batchwise_ranking_regularizer(new_feature, batch_y, 
                    args.interpolation_lambda))
                else:
                    loss += (args.regularization_weight * batchwise_ranking_regularizer(feature, batch_y, 
                    args.interpolation_lambda))
            else:
                    loss += (args.regularization_weight * batchwise_ranking_regularizer(feature, batch_y, 
                    args.interpolation_lambda))

        loss_value = loss.data.cpu().numpy()
        train_batch_losses.append(loss_value)
        loss_list.append(round(loss_value.item(),4))
        ########################################################################
        # optimization step:
        ########################################################################
        optimizer.zero_grad() # (reset gradients)
        loss.backward() # (compute gradients)
        optimizer.step() # (perform optimization step)
        # if args.use_buffer :
        #     encodings.extend(feature.data.squeeze().cpu().numpy())
        #     labels.extend(args.norm * batch_y.data.squeeze().cpu().numpy())

        #     encodings, labels = torch.from_numpy(np.vstack(encodings)).cuda(), torch.from_numpy(np.hstack(labels)).cuda()


        n_step = len(train_loader) // 2
        if (step+1) % n_step ==0:
            print("------------------------------")
            print("Epoch : [{}][{}/{}] , Loss: {:.4f}".format(epoch+1,step+1,len(train_loader),np.mean(train_batch_losses[step + 1 - n_step:step])))
            with open(args.save_file +  "/log_c3d_new_loss.txt","w") as fh:
                fh.write("Epoch "+ str(epoch) + "/" + str(100) + "  Step " +  str(step) + "/" + str(len(train_loader)) + ":" +"\n"\
                                # "mse loss :" + str('%.4f'%mse_loss.item()).replace("'","") + "     nce loss: " + str('%.4f'%loss_nce.item()).replace("'","") + "\n"\
                            + "loss list: " +str(loss_list) + "\n")
                fh.close()


            val_rmse = validation(model,epoch,step,val_loader,temp_gate)
            model.train()
            if args.use_buffer and epoch >= 9:
                encodings, labels = [], []
                with torch.no_grad():
                    for inputs, targets in train_loader:
                        inputs = inputs.cuda(non_blocking=True)
                        inputs=inputs.to(torch.float32)
                        targets=targets.to(torch.float32)
                        inputs=inputs.permute(0,4,1,2,3)
                        feature,output,_ = model(inputs, epoch,args.norm * targets)
                        encodings.extend(feature.data.squeeze().cpu().numpy())
                        labels.extend(args.norm * targets.data.squeeze().cpu().numpy())

                encodings, labels = torch.from_numpy(np.vstack(encodings)).cuda(), torch.from_numpy(np.hstack(labels)).cuda()
                model.FDS.update_last_epoch_stats(epoch)
                model.FDS.update_running_stats(encodings, labels, epoch)
            if val_rmse <= temp_gate:
                temp_gate = val_rmse
    return temp_gate

def validation(model,epoch,step,val_loader,val_gate):
    model.eval()
    # outs = []
    prediction = [] 
    labels = []
    # sigma_list = []
    if args.dataset == "avec2014":
        val_estimate = Estimate_Video('../datasets/val_14.txt')
    else:
        val_estimate = Estimate_Video('../datasets/val_13.txt')
    for batch_x, batch_y in val_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        batch_x=batch_x.permute(0,4,1,2,3)
        batch_x=batch_x.to(torch.float32)
        batch_y=batch_y.to(torch.float32)
        with torch.no_grad():
            if args.feature:
                feature,out = model(batch_x,epoch)
            else:
                out = model(batch_x,epoch)
        prediction.append(args.norm * out)
        labels.append(batch_y)
    prediction = torch.cat(tuple(prediction), 0)
    labels = torch.cat(tuple(labels), 0)
    MAE , RMSE = val_estimate.estimate(prediction , labels)
    print("------------------------------")
    print('val_MAE:{:.4f} , val_RMSE:{:.4f} '.format(MAE.item(),RMSE.item()))
    val_rmse = test(model,test_loader,epoch,step,listMAE,listRMSE,MAE,RMSE,val_gate)
    return val_rmse


def test(model,test_loader,epoch,step,listMAE,listRMSE,val_MAE,val_RMSE,val_gate):
    model.eval()
    prediction = [] 
    labels = []
    if args.dataset == "avec2014":
        test_estimate = Estimate_Video('../datasets/test_video_14.txt')
    else:
        test_estimate = Estimate_Video('../datasets/test_video_13.txt')
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        batch_x=batch_x.permute(0,4,1,2,3)
        batch_x=batch_x.to(torch.float32)
        batch_y=batch_y.to(torch.float32)
        with torch.no_grad():
            if args.feature:
                feature,out = model(batch_x,epoch)
            else:
                out = model(batch_x,epoch)
        prediction.append(args.norm * out)
        labels.append(batch_y)

    prediction = torch.cat(tuple(prediction), 0)
    labels = torch.cat(tuple(labels), 0)
    MAE , RMSE = test_estimate.estimate(prediction , labels)
    print("------------------------------")
    print('test_MAE:{:.4f} , test_RMSE:{:.4f} '.format(MAE.item(),RMSE.item()))
    listRMSE.append(round(RMSE.item(),4))
    listMAE.append(round(MAE.item(),4))
    if val_RMSE <= val_gate:
        torch.save(model.state_dict(), args.save_file  + '/C3D_' +str(epoch) + "_" + str(step) +' val_m' + str(round(val_MAE.item(),2)) + " val_r" + str(round(val_RMSE.item(),2))+' m' + str(round(MAE.item(),2)) + "r" + str(round(RMSE.item(),2))+'.pkl')
    with open(args.save_file  + "/log_c3d_new.txt","w") as fh:
        fh.write("MAE "+str('%.4f'%MAE.item()).replace("'","") + " RMSE " +  str('%.4f'%RMSE.item()).replace("'","") +"\n"\
            +  "listMAE: " +str(listMAE) + "\n"\
            +  "listRMSE: " +str(listRMSE) + "\n")
        fh.close()
    return val_RMSE

if __name__ == '__main__':
    
    if args.dataset == "avec2014":
        train_data = MyDataset(txt='../datasets/train_14.txt', transform=None)
        val_data = MyDataset_test(txt='../datasets/val_14.txt', transform=None)
        test_data = MyDataset_test(txt='../datasets/test_video_14.txt', transform=None)
    else:
        train_data = MyDataset(txt='../datasets/train_13.txt', transform=None)
        val_data = MyDataset_test(txt='../datasets/val_13.txt', transform=None)
        test_data = MyDataset_test(txt='../datasets/test_video_13.txt', transform=None)
    train_loader = DataLoader(dataset=train_data, batch_size=28, shuffle=True,num_workers = 5)
    val_loader = DataLoader(dataset=val_data, batch_size=40,num_workers = 5)
    test_loader = DataLoader(dataset=test_data, batch_size=40,num_workers = 5)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lr = 1e-4

    model=C3D_model_with_features.C3D(use_feature=args.feature,use_buffer = args.use_buffer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr ,weight_decay = 1e-4)
    loss_func = torch.nn.MSELoss()

    model = model.to(device)
    listRMSE = []
    listMAE = []
    loss_list = []

    val_gate = 20
    for epoch in range(0 ,300):
        temp = train(model,train_loader,epoch,optimizer,loss_list,listMAE,listRMSE,val_gate)
        if temp <= val_gate:
            val_gate = temp




