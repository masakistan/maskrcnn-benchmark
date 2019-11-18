import torch
from torch.nn import functional as F
torch.set_printoptions(profile="full")

from maskrcnn_benchmark.modeling.matcher import Matcher

from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler,
)
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist

from maskrcnn_benchmark.structures.keypoint import keypoints_to_heat_map, keypoints_to_heat_map_multi_point


def project_keypoints_to_heatmap(keypoints, proposals, discretization_size, nclasses):
    proposals = proposals.convert("xyxy")
    return keypoints_to_heat_map_multi_point(
        keypoints, proposals.bbox, discretization_size, nclasses
    )


def cat_boxlist_with_keypoints(boxlists):
    assert all(boxlist.has_field("keypoints") for boxlist in boxlists)

    kp = [boxlist.get_field("keypoints").keypoints for boxlist in boxlists]
    kp = cat(kp, 0)

    fields = boxlists[0].get_fields()
    fields = [field for field in fields if field != "keypoints"]

    boxlists = [boxlist.copy_with_fields(fields) for boxlist in boxlists]
    boxlists = cat_boxlist(boxlists)
    boxlists.add_field("keypoints", kp)
    return boxlists


def _within_box(points, boxes):
    """Validate which keypoints are contained inside a given box.
    points: NxKx2
    boxes: Nx4
    output: NxK
    """
    x_within = (points[..., 0] >= boxes[:, 0, None]) & (
        points[..., 0] <= boxes[:, 2, None]
    )
    y_within = (points[..., 1] >= boxes[:, 1, None]) & (
        points[..., 1] <= boxes[:, 3, None]
    )
    return x_within & y_within


class KeypointRCNNLossComputation(object):
    def __init__(self, proposal_matcher, fg_bg_sampler, discretization_size):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            discretization_size (int)
        """
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.discretization_size = discretization_size

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        #print('match quality matrix', match_quality_matrix.shape)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Keypoint RCNN needs "labels" and "keypoints "fields for creating the targets
        target = target.copy_with_fields(["labels", "keypoints"])
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        #print('matched', matched_idxs, matched_idxs.shape)
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        #print('prepare proposals', proposals)
        #print('prepare targets', targets)
        labels = []
        keypoints = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # this can probably be removed, but is left here for clarity
            # and completeness
            # TODO check if this is the right one, as BELOW_THRESHOLD
            neg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[neg_inds] = 0

            #print('')
            keypoints_per_image = matched_targets.get_field("keypoints")
            #print('kps per image', keypoints_per_image)
            within_box = _within_box(
                keypoints_per_image.keypoints, matched_targets.bbox
            )
            #print('within box', within_box.shape)
            vis_kp = keypoints_per_image.keypoints[..., 2] > 0
            is_visible = (within_box & vis_kp).sum(1) > 0

            labels_per_image[~is_visible] = -1

            labels.append(labels_per_image)
            keypoints.append(keypoints_per_image)

        return labels, keypoints

    def subsample(self, proposals, targets):
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])
            targets (list[BoxList])
        """

        labels, keypoints = self.prepare_targets(proposals, targets)
        #print("sub labels", labels)
        #print("sub keypoints", keypoints)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        proposals = list(proposals)
        # add corresponding label and regression_targets information to the bounding boxes
        for labels_per_image, keypoints_per_image, proposals_per_image in zip(
            labels, keypoints, proposals
        ):
            #print('per img', keypoints_per_image)
            proposals_per_image.add_field("labels", labels_per_image)
            proposals_per_image.add_field("keypoints", keypoints_per_image)

        # distributed sampled proposals, that were obtained on all feature maps
        # concatenated via the fg_bg_sampler, into individual feature map levels
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
            zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.nonzero(pos_inds_img).squeeze(1)
            proposals_per_image = proposals[img_idx][img_sampled_inds]
            proposals[img_idx] = proposals_per_image

        self._proposals = proposals
        return proposals

    def __call__(self, proposals, keypoint_logits, obj_logits, coord_logits):
        #print('initial kp logits', keypoint_logits.shape)
        heatmaps = []
        valid = []
        valid_one_hot = []
        coord_heatmaps = []
        #print('logits', keypoint_logits.shape)
        N, K, H, W = keypoint_logits.shape
        #print('kp_logits', keypoint_logits.shape)
        #print('obj_ogits', obj_logits.shape)
        #print('coord_logits', coord_logits.shape)
        for proposals_per_image in proposals:
            kp = proposals_per_image.get_field("keypoints")
            #print('kp', kp)
            #print(kp.keypoints.shape)
            #print(kp.keypoints[:,3:11].shape)
            heatmaps_per_image, coord_heatmaps_per_image, valid_per_image = project_keypoints_to_heatmap(
                kp, proposals_per_image, self.discretization_size, K
            )

            #print(heatmaps_per_image.shape)
            #for ix, a in enumerate(heatmaps_per_image):
            #    show = None
            #    show = cv2.normalize(a[3].cpu().detach().numpy(), show, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            #    show = cv2.applyColorMap(show, cv2.COLORMAP_JET)
            #    cv2.imwrite("{}.jpg".format(ix), show)

            #print('hpi', coord_heatmaps_per_image.shape)

            #print('valid per image', valid_per_image)
            heatmaps.append(heatmaps_per_image.view(-1))
            valid.append(valid_per_image.view(-1))
            valid_one_hot.append(valid_per_image.view(-1))
            coord_heatmaps.append(coord_heatmaps_per_image)

        keypoint_targets = cat(heatmaps, dim=0)
        valid = cat(valid, dim=0).to(dtype=torch.bool)
        valid_one_hot = cat(valid_one_hot, dim=0)
        coord_targets = cat(coord_heatmaps, dim=0)
        #print('keypoint_targets', keypoint_targets.shape)
        #print('coord_targets', coord_targets.shape)
        #print('valid', valid, valid.shape)
        valid = torch.nonzero(valid).squeeze(1)
        #print('valid', valid, valid.shape)

        # torch.mean (in binary_cross_entropy_with_logits) does'nt
        # accept empty tensors, so handle it sepaartely
        if keypoint_targets.numel() == 0 or len(valid) == 0:
            return keypoint_logits.sum() * 0

        keypoint_logits = keypoint_logits.view(N * K, H * W)
        obj_logits = obj_logits.view(N * K, -1)
        #print('coord logits', coord_logits.shape)
        coord_logits = coord_logits.view(N * K, 4, H * W)

        #print('kp_logits', keypoint_logits.shape)
        #print('obj_ogits', obj_logits.shape)
        #print('coord_logits', coord_logits.shape)
        #print('p', coord_logits.shape)
        #print('t', coord_targets.shape)
        #print('valid', valid)
        #print(coord_logits[valid].shape)
        #print(coord_targets[valid].shape)
        coord_targets = coord_targets[valid].view(-1)
        #print('t', coord_targets.shape)
        nrows = coord_targets.shape[0]
        #print('nrows', nrows)
        coord_logits = coord_logits[valid].view(nrows, -1)
        #print('p', coord_logits.shape)
        #vtest = valid[:1]
        #val, idx = coord_targets[vtest].max(dim=0)
        #print('act', F.softmax(coord_activations[vtest], 1))
        #print('act', F.softmax(coord_activations[vtest], 1).max(1))
        #print('valid', coord_activations[valid].shape)
        #print('vtest', vtest)
        #print('axs', val, idx)
        #print('tar', coord_targets[vtest])
        #print('key', keypoint_targets[valid])
        #print(coord_activations[valid].shape, coord_targets[valid].shape)
        #sys.exit()
        #print(coord_activations[valid])
        #print(coord_targets[valid])

        objectness_loss = F.cross_entropy(obj_logits, valid_one_hot)
        keypoint_loss = F.cross_entropy(keypoint_logits[valid], keypoint_targets[valid])
        coord_loss = F.cross_entropy(coord_logits, coord_targets)
        #coord_loss = F.binary_cross_entropy(coord_activations, coord_targets)
        return keypoint_loss, objectness_loss, coord_loss


def make_roi_keypoint_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )
    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
    )
    resolution = cfg.MODEL.ROI_KEYPOINT_HEAD.RESOLUTION
    loss_evaluator = KeypointRCNNLossComputation(matcher, fg_bg_sampler, resolution)
    return loss_evaluator
