import torch

from .roi_keypoint_feature_extractors import make_roi_keypoint_feature_extractor
from .roi_keypoint_predictors import make_roi_keypoint_predictor, make_roi_keypoint_predictor_multi
from .roi_keypoint_predictors_objectness import make_keypoint_objectness_predictor
from .inference import make_roi_keypoint_post_processor
from .loss import make_roi_keypoint_loss_evaluator


class ROIKeypointHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(ROIKeypointHead, self).__init__()
        self.cfg = cfg.clone()
        self.feature_extractor = make_roi_keypoint_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_keypoint_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_keypoint_post_processor(cfg)
        self.loss_evaluator = make_roi_keypoint_loss_evaluator(cfg)

        self.objectness_predictor = make_keypoint_objectness_predictor(
            cfg, self.feature_extractor.out_channels)
        self.coord_predictor = make_roi_keypoint_predictor_multi(
            cfg, self.feature_extractor.out_channels)

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the original proposals
                are returned. During testing, the predicted boxlists are returned
                with the `mask` field set
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        #print('features', features)
        #print('proposals', proposals)
        #print('targets', targets)
        #print('targets before', targets[0].get_field('keypoints').keypoints)
        if self.training:
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)

        #print('proposals subsampled', proposals)
        x = self.feature_extractor(features, proposals)
        #print('x', x.shape)
        kp_logits = self.predictor(x)
        coord_logits = self.coord_predictor(x)
        #rint('preds:', kp_logits.shape)
        obj_logits = self.objectness_predictor(x)
        #print('obj', obj_logits.shape)

        if not self.training:
            result = self.post_processor(kp_logits, obj_logtis, coord_logits, proposals)
            return x, result, {}

        loss_kp, loss_kp_obj, loss_kp_coord = self.loss_evaluator(proposals, kp_logits, obj_logits, coord_logits)

        losses = dict(loss_kp=loss_kp, loss_kp_obj=loss_kp_obj, loss_kp_coord=loss_kp_coord)
        print(losses)

        return x, proposals, losses
    

def build_roi_keypoint_head(cfg, in_channels):
    return ROIKeypointHead(cfg, in_channels)
