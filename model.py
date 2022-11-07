import logging
from queue import Queue, Empty
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models import resnet18
import tensorflow as tf
from tensorflow import keras

class detection:
    def __init__(self, path_model_od, path_model_cp, path_model_ve, path_model_te, path_text, size, patch_size, batchsize):
        # network params
        self.path_model_od = path_model_od
        self.path_model_cp = path_model_cp
        self.path_model_ve = path_model_ve
        self.path_model_te = path_model_te
        self.unique_texts = pd.read_csv(path_text)['explanation'].to_list()
        self.size = size
        self.patch_size = patch_size
        self.batchsize = batchsize
        self.device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
    
    def get_model_od(self):
        # Instantiates object detection model
        # load a model pre-trained on COCO
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        self.num_classes = 2
        # replace the classifier with a new one, that has
        # num_classes which is user-defined
        # 1 class (mitosis) + background (non-mitosis)
        # get number of input features for the classifier
        self.in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(self.in_features, self.num_classes)
        return self.model

    def get_model_cp(self):
        # Instantiates cell patch classifier model
        # load pre-trained resnet18
        self.model = resnet18(pretrained=True)
        self.num_classes = 2
        self.num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_ftrs, self.num_classes)
        input_size = 224
        return self.model

    def load_model_od(self):
        # Loads fine-tuned weights into OD model

        # get pre-trained model
        self.model_od = self.get_model_od()
        self.mean = torch.FloatTensor([0.7441, 0.5223, 0.6265]).to(self.device)  # state['data']['normalize']['mean']
        self.std = torch.FloatTensor([0.1409, 0.1618, 0.1320]).to(self.device)  # state['data']['normalize']['std']

        # load in fine-tuned weights to model and save to device
        if torch.cuda.is_available():
            print("Model loaded on CUDA")
            self.model_od.load_state_dict(torch.load(self.path_model_od))
        else:
            print("Model loaded on CPU")
            self.model_od.load_state_dict(torch.load(self.path_model_od, map_location='cpu'))

        self.model_od.to(self.device)

        logging.info("Model OD loaded. Mean: {} ; Std: {}".format(self.mean, self.std))
        return self.model_od

    def load_model_cp(self):
        # Loads fine-tuned weights into CP model

        # get pre-trained model
        self.model_cp = self.get_model_cp()
        self.mean = torch.FloatTensor([0.7441, 0.5223, 0.6265]).to(self.device)  # state['data']['normalize']['mean']
        self.std = torch.FloatTensor([0.1409, 0.1618, 0.1320]).to(self.device)  # state['data']['normalize']['std']

        # load in fine-tuned weights to model and save to device
        if torch.cuda.is_available():
            print("Model loaded on CUDA")
            self.model_cp.load_state_dict(torch.load(self.path_model_cp))
        else:
            print("Model loaded on CPU")
            self.model_cp.load_state_dict(torch.load(self.path_model_cp, map_location='cpu'))

        self.model_cp.to(self.device)

        logging.info("Model CP loaded. Mean: {} ; Std: {}".format(self.mean, self.std))
        return self.model_cp

    def patches(self, input_image):
        # Take input image and generate coordinates for overlapping crops
        queue_patches = Queue()
        img_dimensions = input_image.shape

        n_patches = 0
        # create overlapping patches for the whole image
        for x in np.arange(0, img_dimensions[1], int(0.9 * self.size)):
            for y in np.arange(0, img_dimensions[0], int(0.9 * self.size)):
                # last patch shall reach just up to the last pixel
                if (x+self.size>img_dimensions[1]):
                    x = img_dimensions[1]-512

                if (y+self.size>img_dimensions[0]):
                    y = img_dimensions[0]-512

                queue_patches.put((0, int(x), int(y), input_image))
                n_patches += 1

        n_batches = int(np.ceil(n_patches / self.batchsize))

        return n_batches, queue_patches
    
    def cell_patches(self, input_image):
        queue_cells = Queue()

        # input image dims
        h = input_image.shape[0]
        w = input_image.shape[1]

        # get detections from OD model
        detections = self.object_detection(input_image)

        n_patches = 0
        for idx, detection in enumerate(detections):
            # coordinates of bbox on the input image
            xmin = detections[idx][0]
            ymin = detections[idx][1]
            xmax = detections[idx][2]
            ymax = detections[idx][3]

            # detection centroid
            x = (xmin + xmax)/2
            y = (ymin + ymax)/2

            # upper left corner
            lu_x = int(x - self.patch_size/2)
            lu_y = int(y - self.patch_size/2)

            # ensure image crops are not out of bounds
            # if they are, will create in-bounds non-centered patch
            # constrain x-coordinates
            if lu_x < 0:
                lu_x = 0
            elif x + self.patch_size/2 > w:
                lu_x = w - self.patch_size
            
            # constrain y-coordinates
            if lu_y < 0:
                lu_y = 0
            elif y + self.patch_size/2 > h:
                lu_y = h - self.patch_size

            queue_cells.put((0, int(lu_x), int(lu_y), input_image))
            n_patches+=1
        n_batches = int(np.ceil(n_patches / self.batchsize))
        return n_batches, queue_cells

    def get_batch(self, queue_patches, size):
        # Take a queue with specified patch size (512 for OD, 250 for CP) 
        # to create PyTorch batch for object detection and cell patch classifier
        batch_images = np.zeros((self.batchsize, 3, size, size))
        batch_x = np.zeros(self.batchsize, dtype=int)
        batch_y = np.zeros(self.batchsize, dtype=int)
        for i_batch in range(self.batchsize):
            if queue_patches.qsize() > 0:
                status, batch_x[i_batch], batch_y[i_batch], image = queue_patches.get()
                x_start, y_start = int(batch_x[i_batch]), int(batch_y[i_batch])

                cur_patch = image[y_start:y_start+size, x_start:x_start+size] / 255.
                batch_images[i_batch] = cur_patch.transpose(2, 0, 1)[0:3]
            else:
                batch_images = batch_images[:i_batch]
                batch_x = batch_x[:i_batch]
                batch_y = batch_y[:i_batch]
                break
        torch_batch = torch.from_numpy(batch_images.astype(np.float32, copy=False)).to(self.device)

        #for p in range(torch_batch.shape[0]):
            #torch_batch[p] = transforms.Normalize(mean, std)(torch_batch[p])
        return torch_batch, batch_x, batch_y
    
    def postprocess_patch(self, cur_bbox_pred, cur_label_pred, cur_score_pred, x_real, y_real):
        # Get the best predicted bounding box and score for one patch
        # take the first bounding box prediction (i.e., highest score)
        # in future could implement non-maxima supression for sophisticated bounding box selection
        if len(cur_bbox_pred) > 0:
            # copy torch tensors to cpu if running CUDA
            best_bbox = cur_bbox_pred[0].cpu()
            best_label = cur_label_pred[0].cpu()
            best_score = cur_score_pred[0].cpu()
            # coordinates of bbox on the patch (lu = (0,0))
            xmin = best_bbox[0]
            ymin = best_bbox[1]
            xmax = best_bbox[2]
            ymax = best_bbox[3]
            # store bbox relative to real lu coordinates on original input image
            cur_patch_box = np.array([xmin + x_real, ymin + y_real, xmax + x_real, ymax + y_real, best_label, best_score])

        else: # if no predictions were made, return empty list
            cur_patch_box = []
        return cur_patch_box


    def object_detection(self, input_image):
        # might load model in main script
        #model_od = load_od_model() # load fine-tuned model
        self.model_od.eval()

        n_batches, queue_patches = self.patches(input_image) # get overlapping patch coordinates

        image_boxes = []
        with torch.no_grad():
            # loop through batches
            for _ in tqdm(range(n_batches), desc='Processing an image'):
                torch_batch, batch_x, batch_y = self.get_batch(queue_patches, self.size)
                # perform transforms
                torch_batch = transforms.Normalize(self.mean, self.std)(torch_batch)
                output = self.model_od(torch_batch)

                # loop through each patch in the batch
                for b in range(torch_batch.shape[0]):
                    # get coordinates of lu on original input image
                    x_real = batch_x[b]
                    y_real = batch_y[b]
                    # get bounding boxes, labels, and scores
                    cur_bbox_pred = output[b]['boxes']
                    cur_label_pred = output[b]['labels']
                    cur_score_pred = output[b]['scores']
                    # get best bounding box
                    cur_patch_box = self.postprocess_patch(cur_bbox_pred, cur_label_pred, cur_score_pred, x_real, y_real)
                    if len(cur_patch_box) > 0:
                        image_boxes.append(cur_patch_box)
        return np.array(image_boxes)

    def patch_classifier(self, input_image):
        # might load model in main script
        #model_cp = load_model_cp() # load fine-tuned model
        self.model_cp.eval()

        n_batches, queue_cells = self.cell_patches(input_image) # get overlapping patch coordinates

        predictions = []
        heatmaps = []
        coordinates = {'x': [],
                        'y': []}

        # loop through batches
        for _ in tqdm(range(n_batches), desc='Processing an image'):
            torch_batch, batch_x, batch_y = self.get_batch(queue_cells, self.patch_size)
            # record coordinates of lu on original/input image
            coordinates['x'] = coordinates['x'] + batch_x.tolist()
            coordinates['y'] = coordinates['y'] + batch_y.tolist()
            # perform transforms
            torch_batch = transforms.Resize(224)(torch_batch)
            torch_batch = transforms.Normalize(self.mean, self.std)(torch_batch)
            # predictions
            with torch.no_grad():
                output = self.model_cp(torch_batch)
                _, preds = torch.max(output, 1)
                predictions = predictions + preds.tolist()

            # get heatmaps
            for idx, img in enumerate(torch_batch):
                heatmap = self.gradcam(img)
                heatmaps.append(heatmap)
            
        return predictions, heatmaps, coordinates

    def gradcam(self, img):
        self.model_cp.eval()
        self.model_cp.zero_grad()

        img = img.cpu() # copy image to cpu if using CUDA
        img = img.unsqueeze(0)


        def __extract(grad):
            global feature_grad
            feature_grad = grad
        # get features from the last convolutional layer
        img = img.to(self.device) # write image to device
        x = self.model_cp.conv1(img)
        x = self.model_cp.bn1(x)
        x = self.model_cp.relu(x)
        x = self.model_cp.maxpool(x)
        x = self.model_cp.layer1(x)
        x = self.model_cp.layer2(x)
        x = self.model_cp.layer3(x)
        x = self.model_cp.layer4(x)
        features = x

        # hook for the gradients
        def __extract_grad(grad):
            global feature_grad
            feature_grad = grad
        features.register_hook(__extract_grad)

        # get the output from the whole VGG architecture
        x = self.model_cp.avgpool(x)
        x = x.view(x.size(0), -1)
        output = self.model_cp.fc(x)
        pred = torch.argmax(output).item()
        #print(pred)

        # get the gradient of the output
        output[:, pred].backward()

        # pool the gradients across the channels
        pooled_grad = torch.mean(feature_grad, dim=[0, 2, 3])

        # weight the channels with the corresponding gradients
        # (L_Grad-CAM = alpha * A)
        features = features.detach()
        for i in range(features.shape[1]):
            features[:, i, :, :] *= pooled_grad[i] 

        # average the channels and create an heatmap
        # ReLU(L_Grad-CAM)
        features = features.cpu() # copy features to cpu if using CUDA
        heatmap = torch.mean(features, dim=1).squeeze()
        heatmap = np.maximum(heatmap, 0)

        # normalization for plotting
        heatmap = heatmap / torch.max(heatmap)
        heatmap = heatmap.numpy()

        return heatmap

    def heatmap(self, input_image, predictions, heatmaps, coordinates):
        mask = np.zeros(input_image.shape)
        for idx, prediction in enumerate(predictions):
            if prediction == 0:
                # upper left corner of cell patch/heatmap
                x_start = coordinates['x'][idx]
                y_start = coordinates['y'][idx]
                
                # heatmap
                heatmap = heatmaps[idx]
                heatmap = cv2.resize(heatmap, (250, 250))
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

                mask[y_start:y_start+250, x_start:x_start+250, :] = heatmap

        superimposed_img = input_image + mask
        superimposed_img = np.uint8(255 * superimposed_img / np.max(superimposed_img))
        superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
        return superimposed_img

    def img_resize(self, input_image):
        img_array = tf.image.decode_jpeg(tf.io.read_file(input_image), channels=3)
        resized_img = tf.image.resize(img_array, (299, 299))
        return resized_img

    def load_model_ve(self):
        self.vision_encoder = keras.models.load_model(self.path_model_ve())
        return self.vision_encoder

    def load_model_te(self):
        self.text_encoder = keras.models.load_model(self.path_model_te())
        return self.text_encoder

    def get_img_embedding(self, input_image):
        embedding = self.vision_encoder.predict(
            tf.expand_dims(self.img_resize(input_image), axis=0),
            verbose=0,
        )
        return embedding

    def get_text_embedding(self):
        text_embeddings = [
            self.text_encoder(tf.convert_to_tensor([text.lower().replace('.', '')])) for text in self.unique_texts
        ]
        return text_embeddings

    def text_explanation(self, input_image, k=1, normalize=True):
        get_image_embedding = self.get_img_embedding(input_image)
        get_text_embedding = self.get_text_embedding()
        # Normalize text and image embedding
        if normalize:
            image_embedding = tf.math.l2_normalize(get_image_embedding)
            text_embedding = [tf.math.l2_normalize(text) for text in get_text_embedding]
        # Obtain dot similarities
        dot_similarity = [
            dot[0][0].numpy() for dot in tf.matmul(image_embedding, text_embedding, transpose_b=True)
        ]
        # Retrieve top index similarity score
        result_index = dot_similarity.index(max(dot_similarity))
        # Return best explanation sentence match
        best_explanation = self.unique_texts[result_index]
        return best_explanation
