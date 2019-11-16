import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark import layers
from maskrcnn_benchmark.modeling import registry

#@registry.ROI_KEYPOINT_PREDICTOR_OBJECTNESS.register("KeypointObjectnessPredictor")
class KeypointObjectnessPredictor(nn.Module):
    def __init__(self, cfg, in_channels):
        super(KeypointObjectnessPredictor, self).__init__()

        input_features = in_channels
        self.num_keypoints = cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_CLASSES
        self.conv1_kps_obj = layers.Conv2d(input_features, input_features * 2, 5, 2)
        self.pool1_kps_obj = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_kps_obj = nn.Linear(input_features * 2, self.num_keypoints * 2)
        #self.fc_kps_coords = nn.Linear(input_features * 2, self.num_keypoints * 8)

        nn.init.kaiming_normal_(self.conv1_kps_obj.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.conv1_kps_obj.bias, 0)

    def forward(self, x):
        N = x.shape[0]
        x = self.conv1_kps_obj(x)
        x = F.relu(x)
        x = self.pool1_kps_obj(x)
        #print('x shape', x.shape)
        x = torch.flatten(x, 1)
        #print('x shape', x.shape)
        obj = self.fc_kps_obj(x).view(N, self.num_keypoints, 2)
        #print('obj shape', obj.shape)
        #coords = self.fc_kps_coords(x).view(N, self.num_keypoints, 8)
        #coords = F.relu(coords)
        #return obj, coords
        return obj

def make_keypoint_objectness_predictor(cfg, in_channels):
    return KeypointObjectnessPredictor(cfg, in_channels)
