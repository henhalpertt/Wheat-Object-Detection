import os
import json
import pandas as pd
import numpy as np
from PIL import Image
import logging
# import boto3 
# I wanted to retreive one image from the bucket for inference. but reading the docs I see there are restrictions in doing that.
# so I tried to use boto3 and it didnt even read the script. (ModelError then runtime error)
# so I will shut down the deserializer, and create my own. predictor.predict(input) will get <byte class> and 
# a custom deserializer in input_fn() will convert it: <byte class> --> image with shape [3, 1024, 1024]
import io

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms

logger = logging.getLogger(__name__)

def model_fn(model_dir):
    print("model loading ")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
    input_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(input_features, 2)
    
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))
    logger.info('state.dict loaded into model')

    model.to(device).eval()
    
    logger.info('Done loading model')
    return model

# data preprocessing --> request body is json that contains an URL to some image. example {"url": test_image.jpg}
def input_fn(request_body, content_type='image/jpeg'):
    logger.info('Deserializing the input data..')
    print("type of content: ", content_type)
    print("type of req_body:", type(request_body))
    
    if content_type=='image/jpeg':
        test_image = Image.open(io.BytesIO(request_body)).convert("RGB")
        # tensor transform: 
        logger.info('image was converted to rgb format. starting transformation')
        image_transform = transforms.Compose([transforms.ToTensor()])
        print("shape of img after trans:", [image_transform(test_image)][0].shape)
        return [image_transform(test_image)] # the model(input) needs a list of tensors     
    raise Exception(f'Requested unsupported ContentType in content_type {content_type}')

# inference --> input_object is a tensor. function retrieves it from input_fn return value.  
# input_object is a list of tensors. since I am only predicting for 1 image, its value is input_object[0].
def predict_fn(input_object, model):
    logger.info('Generating prediction from input parameters...')

    image = list(image.to('cuda') for image in input_object)
    with torch.no_grad():
        model.eval() #just to make sure.
        # model is already on gpu + on eval mode (defined in model_fn), but I need to transfer inputs to cuda as well.  
        try:
            print("trying")
            prediction = model(list(image.to('cuda') for image in input_object))
        except:
            print("exception when using model. Please pay close attention to format")
        
    return prediction

# output back to local dir --> predictions retrieved from predict_fn return value. 
def output_fn(predictions, content_type='application/json'):

    print("type of content in output: ", content_type)
    logger.info('Serializing the output now..')
    score_threshold = 0.5
    image_outputs = []

    boxes = predictions[0]['boxes'].data.cpu().numpy()
    scores = predictions[0]['scores'].data.cpu().numpy()
    mask = scores >= score_threshold
    
    boxes = boxes[mask].astype(np.int32).tolist()
    scores = scores[mask].tolist()
    image_outputs.append((boxes, scores))
    
    print("values: ", image_outputs)
    return json.dumps(image_outputs)

'''

    Faster R-CNN information: 
    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes. --> .convert("RGB") gives [ , , ] format. 
    
    The behavior of the model changes depending if it is in training or evaluation mode.
    
    ** During training ** the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box
    The model returns a Dict[Tensor] during training, containing the classification and regression
    losses for both the RPN and the R-CNN.
    
    ** During inference** the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores or each prediction
        
'''