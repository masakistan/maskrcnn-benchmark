import torch
from torch import nn
import torch.nn.functional as F

from maskrcnn_benchmark.modeling.utils import cat


class KeypointPostProcessor(nn.Module):
    def __init__(self, keypointer=None):
        super(KeypointPostProcessor, self).__init__()
        self.keypointer = keypointer

    def forward(self, x, obj_logits, coord_logits, boxes):
        mask_prob = x
        N, K, H, W = x.shape
        keypoint_logits = x.view(N * K, H * W)
        keypoint_probs, keypoint_prob_idxs = torch.max(nn.functional.softmax(keypoint_logits, dim=1), 1)
        #coord_logits = coord_logits.view(N, K * 4, H, W)
        #print('coord logits', coord_logits)
        coord_logits = coord_logits.split(1, dim=2)
        #print('coord', coord_logits[0][0][0][0])
        #print('coord', coord_logits[0][0][1][0])
        #print('key', keypoint_logits.shape)
        #print('coo', [y.shape for y in coord_logits])
        #print('x', x.shape)

        scores = None
        if self.keypointer:
            #print('keypointing')
            mask_prob, scores = self.keypointer(x, boxes)
            corners = [self.keypointer(y.view(N, K, H, W), boxes) for y in coord_logits]
            #print([x[0][0] for x in corners])
            #print([x[0][...,:2] for x in corners])
            mask_prob = cat([mask_prob] + [x[0][...,:2] for x in corners], dim = 2)
            #print('new mask prob', mask_prob.shape, mask_prob[0][0], mask_prob[0][1])

        #print('mask prob', mask_prob)
        assert len(boxes) == 1, "Only non-batched inference supported for now"
        boxes_per_image = [box.bbox.size(0) for box in boxes]
        print("boxes per img", boxes_per_image)
        #print('boxes per image', boxes_per_image)
        #print(mask_prob.shape)
        mask_prob = mask_prob.split(boxes_per_image, dim=0)
        scores = scores.split(boxes_per_image, dim=0)
        #print([x.shape for x in mask_prob])
        #print([x[0].shape for x in corners])
        #corner_mask_prob = cat([x[0].shape])
        obj_probs = F.softmax(obj_logits, dim=2)
        print('obj probs', obj_probs.shape)
        max_vals, max_idxs = obj_logits.max(dim=2)
        visible = obj_probs[..., 1]
        print('visible', visible.shape)
        #print('obj logits', obj_logits, max_idxs)
        #print(visible)

        results = []
        for prob, box, score in zip(mask_prob, boxes, scores):
            bbox = BoxList(box.bbox, box.size, mode="xyxy")
            for field in box.fields():
                bbox.add_field(field, box.get_field(field))
            #print('prob', prob.shape)
            prob = PersonKeypoints(prob, box.size, [0, K])
            prob.add_field("logits", score)
            bbox.add_field("keypoints", prob)
            bbox.add_field("keypoint_probs", keypoint_probs)
            bbox.add_field("keypoint_prob_idxs", keypoint_prob_idxs)
            bbox.add_field("obj_probs", visible)
            #print("keypoint probs", keypoint_probs, keypoint_probs.shape)
            #print("keypoint idxs", keypoint_prob_idxs, keypoint_prob_idxs.shape)
            results.append(bbox)

        return results


# TODO remove and use only the Keypointer
import numpy as np
import cv2


def heatmaps_to_keypoints(maps, rois):
    """Extract predicted keypoint locations from heatmaps. Output has shape
    (#rois, 4, #keypoints) with the 4 rows corresponding to (x, y, logit, prob)
    for each keypoint.
    """
    # This function converts a discrete image coordinate in a HEATMAP_SIZE x
    # HEATMAP_SIZE image to a continuous keypoint coordinate. We maintain
    # consistency with keypoints_to_heatmap_labels by using the conversion from
    # Heckbert 1990: c = d + 0.5, where d is a discrete coordinate and c is a
    # continuous coordinate.
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]

    widths = rois[:, 2] - rois[:, 0]
    heights = rois[:, 3] - rois[:, 1]
    widths = np.maximum(widths, 1)
    heights = np.maximum(heights, 1)
    widths_ceil = np.ceil(widths)
    heights_ceil = np.ceil(heights)

    # NCHW to NHWC for use with OpenCV
    maps = np.transpose(maps, [0, 2, 3, 1])
    print('maps', maps.shape)
    min_size = 0  # cfg.KRCNN.INFERENCE_MIN_SIZE
    num_keypoints = maps.shape[3]
    xy_preds = np.zeros((len(rois), 3, num_keypoints), dtype=np.float32)
    end_scores = np.zeros((len(rois), num_keypoints), dtype=np.float32)
    for i in range(len(rois)):
        if min_size > 0:
            roi_map_width = int(np.maximum(widths_ceil[i], min_size))
            roi_map_height = int(np.maximum(heights_ceil[i], min_size))
        else:
            roi_map_width = widths_ceil[i]
            roi_map_height = heights_ceil[i]
        width_correction = widths[i] / roi_map_width
        height_correction = heights[i] / roi_map_height
        roi_map = cv2.resize(
            maps[i], (roi_map_width, roi_map_height), interpolation=cv2.INTER_CUBIC
        )
        # Bring back to CHW
        roi_map = np.transpose(roi_map, [2, 0, 1])
        # roi_map_probs = scores_to_probs(roi_map.copy())
        print('roimap', roi_map.shape)
        #for ix, a in enumerate(roi_map):
        #    show = None
        #    show = cv2.normalize(a, show, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        #    show = cv2.applyColorMap(show, cv2.COLORMAP_JET)
        #    cv2.imwrite("{}.jpg".format(ix), show)
        w = roi_map.shape[2]
        pos = roi_map.reshape(num_keypoints, -1).argmax(axis=1)
        print('pos:', pos, w)
        x_int = pos % w
        y_int = (pos - x_int) // w

        print(x_int)
        print(y_int)
        # assert (roi_map_probs[k, y_int, x_int] ==
        #         roi_map_probs[k, :, :].max())
        x = (x_int + 0.5) * width_correction
        y = (y_int + 0.5) * height_correction
        xy_preds[i, 0, :] = x + offset_x[i]
        xy_preds[i, 1, :] = y + offset_y[i]
        xy_preds[i, 2, :] = 1
        end_scores[i, :] = roi_map[np.arange(num_keypoints), y_int, x_int]

    return np.transpose(xy_preds, [0, 2, 1]), end_scores


from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints


class Keypointer(object):
    """
    Projects a set of masks in an image on the locations
    specified by the bounding boxes
    """

    def __init__(self, padding=0):
        self.padding = padding

    def __call__(self, masks, boxes):
        # TODO do this properly
        if isinstance(boxes, BoxList):
            boxes = [boxes]
        assert len(boxes) == 1

        result, scores = heatmaps_to_keypoints(
            masks.detach().cpu().numpy(), boxes[0].bbox.cpu().numpy()
        )
        return torch.from_numpy(result).to(masks.device), torch.as_tensor(scores, device=masks.device)


def make_roi_keypoint_post_processor(cfg):
    keypointer = Keypointer()
    keypoint_post_processor = KeypointPostProcessor(keypointer)
    return keypoint_post_processor
