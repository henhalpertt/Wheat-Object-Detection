import argparse
import sys
import os
import json
import pandas as pd
import numpy as np
from PIL import Image
from smart_open import open
import pickle
import time
import boto3

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import Dataset
from torchvision import transforms

# GLOBALS
# bucket name : sagemaker-us-east-1-267252719644 
CSV_FILE_PATH = 'S3://sagemaker-us-east-1-267252719644/wheat-data/train.csv'
DATA_FOLDER_PATH = 's3://sagemaker-us-east-1-267252719644/wheat-data/train/'
IMG_NAME_PATH = 'S3://sagemaker-us-east-1-267252719644/wheat-data/images_names.txt'


class Wheat(Dataset):

    def __init__(self, dataframe, image_dir, transforms=False):
        super().__init__()
        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        records = self.df[self.df['image_id'] == image_id]
        
        img_path = os.path.join(self.image_dir, f"{self.image_ids[idx]}.jpg")
        img = Image.open(open(img_path, 'rb')).convert("RGB")
        boxes = records[['x', 'y', 'w', 'h']].values

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        
        # convert all to tensors:
        image_id = torch.tensor([idx]).permute(1,2,0).cpu().numpy()
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # detecting only one label
        labels = torch.ones((records.shape[0],), dtype=torch.int64)
        # 0 - not crowd
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms:
            train_transforms = transforms.Compose([transforms.RandomHorizontalFlip()])
            img = train_transforms(img)
            
        general_trans = transforms.Compose([transforms.ToTensor()])
        img = general_trans(img)
        
        return img, target, image_id

    def __len__(self) -> int:
        return len(self.image_ids)
    
    
def set_model(num_classes:int):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    input_features = model.roi_heads.box_predictor.cls_score.in_features
    turning off grads for pretrained weights 
    for param in model.parameters():
        param.requires_grad = False
    # changing the last layer according to the dataset:
    model.roi_heads.box_predictor = FastRCNNPredictor(input_features, num_classes)
    return model

def collate_fn(batch):
    return tuple(zip(*batch))

def train_model(train, valid):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    num_classes = 2 # either its wheat object(class=1) or background(class=0)
                              
    train_dataset = Wheat(dataframe=train, image_dir=DATA_FOLDER_PATH, transforms=True)
    val_dataset = Wheat(dataframe=valid, image_dir=DATA_FOLDER_PATH)
    
    # dataloaders:
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=0,pin_memory=True,
                                              collate_fn=collate_fn)
    
    validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=0,pin_memory=True,
                                              collate_fn=collate_fn)
    
    # get the model:
    model = set_model(num_classes=2)
    model.to(device)
    
    #init optimizer object:
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    
    num_epochs = args.epochs
    itr = 1
    
    total_train_loss = []
    total_valid_loss = []

    losses_value = 0

    for epoch in range(num_epochs):
        start_time = time.time()
        print("++++++++++++++++++epoch started++++++++++++++++++")
        
      # ------------------------training ------------------------------
        model.train()
        train_loss = []

        for images, targets, image_ids in train_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            losses_value = losses.item()
            train_loss.append(losses_value)   
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            itr += 1
        epoch_train_loss = np.mean(train_loss)
        total_train_loss.append(epoch_train_loss)
        
        # ------------------------validation ------------------------------

        
        with torch.no_grad():
            valid_loss = []
            
            for images, targets, image_ids in validation_loader:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                loss_value = losses.item()
                valid_loss.append(loss_value)
            
        epoch_valid_loss = np.mean(valid_loss)
        total_valid_loss.append(epoch_valid_loss)
        
        
        print(f"Epoch Completed: {epoch+1}/{num_epochs}, Time: {time.time()-start_time}\n" 
              f"Train Loss: {epoch_train_loss}, Valid Loss: {epoch_valid_loss}\n")
        
    # In addition to state_dict, I also want to save the losses from train/valid and plot them later:
    loss_dict = {
        'total_train_loss': total_train_loss,
        'total_valid_loss':total_valid_loss
    }
    save_model(model, loss_dict)
    return

def save_model(model, loss_dict):
    print("saving trained model...")
            
    model_path = os.path.join(args.model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)
    print("model saved")
    
    # losses: 
    model_loss_path = os.path.join(args.model_dir, 'model_loss.pth')
    with open(model_loss_path, 'wb') as f:
        torch.save(loss_dict, f)
    print("losses saved")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])
    
    # optional training Parameters
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    args = parser.parse_args()
    
    # set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        
    # load csv file, clean it, and break bbox into 4 columns: [x, y, width, height]
    df = pd.read_csv(open(CSV_FILE_PATH), delimiter=',')
    print(df.head(5))
    df['x'] = np.zeros(df.shape[0])
    df['y'] = np.zeros(df.shape[0])
    df['w'] = np.zeros(df.shape[0])
    df['h'] = np.zeros(df.shape[0])
    df[['x','y','w','h']] = np.stack(df['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=','))).astype(np.float)
    df.drop(columns=['bbox'], inplace=True)
    
    # make a df that has only the relevant entries for the images I use for this demo:
    
    #images_names = os.listdir(open(IMG_NAME_PATH))
    
    with open(IMG_NAME_PATH, "r") as fp:   # un-json it
        images_names = json.load(fp)
    
    file_image_ids = []
    for name in images_names:
        image_id = name.split('.jpg')[0]
        file_image_ids.append(image_id)
    df.index = df['image_id']
    demo_data = df[df['image_id'].isin(file_image_ids)]
    
    # split data into validation / train sets:
    image_ids = demo_data['image_id'].unique()
    # 90% will go for training
    split_len = round(len(image_ids)*0.9)
    train_ids = image_ids[:split_len]
    valid_ids = image_ids[split_len:]
    train = demo_data[demo_data['image_id'].isin(train_ids)]
    valid = demo_data[demo_data['image_id'].isin(valid_ids)]
    
    print(f"size of train set: {train['image_id'].unique().shape}\nsize of validation set:{valid['image_id'].unique().shape}")
    # call train

    train_model(train, valid)