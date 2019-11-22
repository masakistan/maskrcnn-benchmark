# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import cv2
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark import layers as L
from maskrcnn_benchmark.utils import cv2_util
torch.set_printoptions(profile="full")

class Resize(object):
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = self.min_size
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        return image
class COCODemo(object):
    # COCO categories for pretty print
    CATEGORIES = [
        "null",
        "_background_",
        "name_col_field",
        "name_col_header",
        "name_col",
        "occupation_col_header",
        "occupation_col_occupation_field",
        "occupation_col_industry_field",
        "occupation_col",
        "veteran_col_header",
        "veteran_col_yes_or_no",
        "veteran_col_war_or_expedition",
        "veteran_col",
        "medcert",
        "cod",
        "contrib",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]

    def __init__(
        self,
        cfg,
        confidence_threshold=0.7,
        show_mask_heatmaps=False,
        masks_per_dim=2,
        min_image_size=224,
        weight_loading = None,
        obj_confidence_threshold=0.7    
    ):
        self.cfg = cfg.clone()
        self.model = build_detection_model(cfg)
        self.model.eval()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)
        #self.min_image_size = min_image_size

        save_dir = cfg.OUTPUT_DIR
        checkpointer = DetectronCheckpointer(cfg, self.model, save_dir=save_dir)
        _ = checkpointer.load(cfg.MODEL.WEIGHT)
        #_ = checkpointer.load()
        
        if weight_loading:
            print('Loading weight from {}.'.format(weight_loading))
            _ = checkpointer._load_model(torch.load(weight_loading))
        
        self.transforms = self.build_transform()

        mask_threshold = -1 if show_mask_heatmaps else 0.5
        self.masker = Masker(threshold=mask_threshold, padding=1)

        # used to make colors for each class
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

        self.cpu_device = torch.device("cpu")
        self.confidence_threshold = confidence_threshold
        self.obj_confidence_threshold = obj_confidence_threshold
        #print('confidence', self.confidence_threshold)
        self.show_mask_heatmaps = show_mask_heatmaps
        self.masks_per_dim = masks_per_dim

    def build_transform(self):
        """
        Creates a basic transformation that was used to train the models
        """
        cfg = self.cfg

        # we are loading images with OpenCV, so we don't need to convert them
        # to BGR, they are already! So all we need to do is to normalize
        # by 255 if we want to convert to BGR255 format, or flip the channels
        # if we want it to be in RGB in [0-1] range.
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        #print(max_size)
        transform = T.Compose(
            [
                T.ToPILImage(),
                Resize(min_size, max_size),
                T.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform

    def run_on_opencv_image(self, image):
        """
        Arguments:
            image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        #print(image.shape)
        predictions = self.compute_prediction(image)
        #print('init prds', len(predictions))
        top_predictions = self.select_top_predictions(predictions)
        #print('top preds', len(top_predictions))
        
        result = image.copy()
        if self.show_mask_heatmaps:
            #pritn('adding heatmap')
            return self.create_mask_montage(result, top_predictions)
        result = self.overlay_boxes(result, top_predictions)
        if self.cfg.MODEL.MASK_ON:
            #print('adding mask')
            result = self.overlay_mask(result, top_predictions)
        if self.cfg.MODEL.KEYPOINT_ON:
            #print('adding keypoint')
            result = self.overlay_keypoints(result, top_predictions)
        result = self.overlay_class_names(result, top_predictions)

        #print('return top preds', len(top_predictions))
        return top_predictions, result

    def compute_prediction(self, original_image):
        """
        Arguments:
            original_image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        # apply pre-processing to image
        image = self.transforms(original_image)
        #print(image.shape)
        # convert to an ImageList, padded so that it is divisible by
        # cfg.DATALOADER.SIZE_DIVISIBILITY
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        # compute predictions
        with torch.no_grad():
            predictions = self.model(image_list)
        predictions = [o.to(self.cpu_device) for o in predictions]

        # always single image is passed at a time
        prediction = predictions[0]
        print('n preds', len(prediction))

        # reshape prediction (a BoxList) into the original image size
        height, width = original_image.shape[:-1]
        prediction = prediction.resize((width, height))

        if prediction.has_field("mask"):
            # if we have masks, paste the masks in the right position
            # in the image, as defined by the bounding boxes
            masks = prediction.get_field("mask")
            # always single image is passed at a time
            masks = self.masker([masks], [prediction])[0]
            prediction.add_field("mask", masks)
        return prediction

    def select_top_predictions(self, predictions):
        """
        Select only predictions which have a `score` > self.confidence_threshold,
        and returns the predictions in descending order of score

        Arguments:
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores`.

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        scores = predictions.get_field("scores")
        print('scores', scores)
        #print(scores, scores > self.confidence_threshold)
        keep = torch.nonzero(scores > self.confidence_threshold).squeeze(1)
        print('keep', keep)
        predictions = predictions[keep]
        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        #if predictions.has_field("obj_probs"):
            #print('modify probs', predictions.get_field("obj_probs").shape)
            #predictions.add_field("obj_probs", predictions.get_field("obj_probs")[idx])
            #print('modify probs', predictions.get_field("obj_probs").shape)
        return predictions[idx]

    def compute_colors_for_labels(self, labels):
        """
        Simple function that adds fixed colors depending on the class
        """
        colors = labels[:, None] * self.palette
        colors = (colors % 255).numpy().astype("uint8")
        return colors

    def overlay_boxes(self, image, predictions):
        """
        Adds the predicted boxes on top of the image

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `labels`.
        """
        labels = predictions.get_field("labels")
        boxes = predictions.bbox
        #print(labels)

        #colors = self.compute_colors_for_labels(labels).tolist()
        labels = torch.from_numpy(np.array([i for i in range(len(labels))]))
        #print(labels)
        colors = self.compute_colors_for_labels(labels).tolist()

        for box, color in zip(boxes, colors):
            box = box.to(torch.int64)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            image = cv2.rectangle(
                image, tuple(top_left), tuple(bottom_right), tuple(color), 2
            )

        return image

    def overlay_mask(self, image, predictions):
        """
        Adds the instances contours for each predicted object.
        Each label has a different color.

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `mask` and `labels`.
        """
        masks = predictions.get_field("mask").numpy()
        labels = predictions.get_field("labels")
        labels = torch.from_numpy(np.array([i for i in range(len(labels))]))

        colors = self.compute_colors_for_labels(labels).tolist()
        #print('total masks', len(masks), len(colors))

        for mask, color in zip(masks, colors):
            thresh = mask[0, :, :, None].astype(np.uint8)
            contours, hierarchy = cv2_util.findContours(
                thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            image = cv2.drawContours(image, contours, -1, color, 3)

        composite = image

        return composite

    def overlay_keypoints(self, image, predictions):
        keypoints = predictions.get_field("keypoints")
        labels = predictions.get_field("labels")
        print('labels', labels)
        #print('keypoints', keypoints)
        kps = keypoints.keypoints
        #print('all kps', kps.shape)
        obj_probs = predictions.get_field("obj_probs")
        probs = predictions.get_field("keypoint_probs")
        prob_idxs = predictions.get_field("keypoint_prob_idxs")
        #print('squeeze', obj_probs.shape)
        print('obj probs', obj_probs.shape)
        #obj_probs = torch.squeeze(obj_probs, 0)
        print('obj probs', obj_probs.shape)
        #print('squeeze', obj_probs.shape)
        #print('probs', probs)
        #print('idxs', prob_idxs)
        #print('kps', kps, kps.shape)
        #print('scores', scores.shape)
        for i, (region, scores, label) in enumerate(zip(keypoints, obj_probs, labels)):
            kps = region.keypoints
            print('\t', scores)
            #print('orig kps', kps.shape, scores.shape)
            #print('\t', i, kps.shape)
            #kps = torch.unsqueeze(kps, 0)
            kps = torch.cat((kps[:, 0:2], kps[:, 3:11]), dim=1).numpy()
            #print('kps', kps.shape)
            image = vis_keypoints(image, kps.transpose((1, 0)), scores, self.obj_confidence_threshold, label)
        return image

    def create_mask_montage(self, image, predictions):
        """
        Create a montage showing the probability heatmaps for each one one of the
        detected objects

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `mask`.
        """
        masks = predictions.get_field("mask")
        masks_per_dim = self.masks_per_dim
        masks = L.interpolate(
            masks.float(), scale_factor=1 / masks_per_dim
        ).byte()
        height, width = masks.shape[-2:]
        max_masks = masks_per_dim ** 2
        masks = masks[:max_masks]
        # handle case where we have less detections than max_masks
        if len(masks) < max_masks:
            masks_padded = torch.zeros(max_masks, 1, height, width, dtype=torch.uint8)
            masks_padded[: len(masks)] = masks
            masks = masks_padded
        masks = masks.reshape(masks_per_dim, masks_per_dim, height, width)
        result = torch.zeros(
            (masks_per_dim * height, masks_per_dim * width), dtype=torch.uint8
        )
        for y in range(masks_per_dim):
            start_y = y * height
            end_y = (y + 1) * height
            for x in range(masks_per_dim):
                start_x = x * width
                end_x = (x + 1) * width
                result[start_y:end_y, start_x:end_x] = masks[y, x]
        return cv2.applyColorMap(result.numpy(), cv2.COLORMAP_JET)

    def overlay_class_names(self, image, predictions):
        """
        Adds detected class names and scores in the positions defined by the
        top-left corner of the predicted bounding box

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores` and `labels`.
        """
        scores = predictions.get_field("scores").tolist()
        labels = predictions.get_field("labels").tolist()
        labels = [self.CATEGORIES[i] for i in labels]
        boxes = predictions.bbox

        template = "{}: {:.2f}"
        for box, score, label in zip(boxes, scores, labels):
            x, y = box[:2]
            s = template.format(label, score)
            cv2.putText(
                image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
            )

        return image

import numpy as np
import matplotlib.pyplot as plt
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints
from maskrcnn_benchmark.structures.census_keypoint_names import keypoint_names, skeletons, keypoint_offsets

def vis_keypoints(img, kps, probs, obj_confidence_threshold, label, kp_thresh=2, alpha=0.7):
    """Visualizes keypoints (adapted from vis_one_image).
    kps has shape (4, #keypoints) where 4 rows are (x, y, logit, prob).
    """
    label = COCODemo.CATEGORIES[label]
    print('vis object type:', label)
    print('vis probs', probs)
    print(kps.shape)
    #kps = kps[probs > obj_confidence_threshold]
    dataset_keypoints = keypoint_names[label]
    kp_lines = skeletons[label]
    offset = keypoint_offsets[label]
    print('vis offset', offset)
    print(kp_lines)

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    #print('kps:', kps.shape)
    #print('scores:', scores)

    for l in range(len(kp_lines)):
        i1 = kp_lines[l][0] - 1 + offset
        i2 = kp_lines[l][1] - 1 + offset
        p1 = kps[0, i1], kps[1, i1]
        p1c1 = kps[2, i1], kps[3, i1]
        p1c2 = kps[4, i1], kps[5, i1]
        p1c3 = kps[6, i1], kps[7, i1]
        p1c4 = kps[8, i1], kps[9, i1]
        
        p2 = kps[0, i2], kps[1, i2]
        p2c1 = kps[2, i2], kps[3, i2]
        p2c2 = kps[4, i2], kps[5, i2]
        p2c3 = kps[6, i2], kps[7, i2]
        p2c4 = kps[8, i2], kps[9, i2]

        print('\t', l)
        print('\t\t', i1)
        print('\t\t\t', p1)
        print('\t\t\t', probs[i1])
        print('\t\t', i2)
        print('\t\t\t', p2)
        print('\t\t\t', probs[i2])
         
        
        #print('point', p1)
        #print('\t', p1c1, p1c2, p1c3, p1c4)
        if probs[i1] > obj_confidence_threshold and probs[i2] > obj_confidence_threshold:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
            
        if probs[i1] > obj_confidence_threshold:
            cv2.line(
                kp_mask, p1c1, p1c2,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
            cv2.line(
                kp_mask, p1c2, p1c3,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
            cv2.line(
                kp_mask, p1c3, p1c4,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
            cv2.line(
                kp_mask, p1c4, p1c1,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
            
            cv2.circle(
                kp_mask, p1c1,
                radius=6, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(
                kp_mask, p1c2,
                radius=6, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(
                kp_mask, p1c3,
                radius=6, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(
                kp_mask, p1c4,
                radius=6, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(
                kp_mask, p1,
                radius=6, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if probs[i2] > obj_confidence_threshold:
            cv2.line(
                kp_mask, p2c1, p2c2,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
            cv2.line(
                kp_mask, p2c2, p2c3,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
            cv2.line(
                kp_mask, p2c3, p2c4,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
            cv2.line(
                kp_mask, p2c4, p2c1,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)

            cv2.circle(
                kp_mask, p2c1,
                radius=6, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(
                kp_mask, p2c2,
                radius=6, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(
                kp_mask, p2c3,
                radius=6, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(
                kp_mask, p2c4,
                radius=6, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(
                kp_mask, p2,
                radius=6, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)
