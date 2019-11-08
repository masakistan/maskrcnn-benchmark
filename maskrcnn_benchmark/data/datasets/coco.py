# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import defaultdict

import torch
import torchvision

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints
from maskrcnn_benchmark.structures.census import CensusNameColKeypoints
from maskrcnn_benchmark.structures.census import CensusOccupationColKeypoints
from maskrcnn_benchmark.structures.census import CensusVeteranColKeypoints


min_keypoints_per_image = 10


def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False


class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None
    ):
        super(COCODataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids

        self.categories = {cat['id']: cat['name'] for cat in self.coco.cats.values()}

        if 'name_col' in self.categories and 'keypoints' in self.coco.cats.values()[0]:
            self.rev_categories = {cat['name']: cat['id'] for cat in self.coco.cats.values()}
            self.keypoints = {cat['id']: cat['keypoints'] for cat in self.coco.cats.values()}
            self.skeletons = {cat['id']: cat['skeleton'] for cat in self.coco.cats.values()}

            CensusNameColKeypoints.CONNECTIONS = self.skeletons[self.rev_categories['name_col']]
            CensusOccupationColKeypoints.CONNECTIONS = self.skeletons[self.rev_categories['occupation_col']]
            CensusVeteranColKeypoints.CONNECTIONS = self.skeletons[self.rev_categories['veteran_col']]

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms

    def __getitem__(self, idx):
        img, anno = super(COCODataset, self).__getitem__(idx)

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        if anno and "segmentation" in anno[0]:
            masks = [obj["segmentation"] for obj in anno]
            masks = SegmentationMask(masks, img.size, mode='poly')
            target.add_field("masks", masks)

        if anno and "keypoints" in anno[0]:
            # NOTE: need to padd keypoints
            lens = [len(x["keypoints"]) for x in anno]
            max_len = max(lens)
            #print('max len', max_len)
            #keypoints = [obj["keypoints"] for obj in anno]
            keypoints = defaultdict(list)
            for x in anno:
                keypoint = x["keypoints"]
                cat_id = x["category_id"]
                keypoints[cat_id].append(keypoint)
                
                #diff = max_len - len(keypoint)
                #keypoints.append(keypoint + ([0] * diff))

            '''
            nam_kps = None
            occ_kps = None
            vet_kps = None
            keypoints = {}
            for cat_id, keypoint in keypoints.items():
                label = self.categories[cat_id]
                if label == 'name_col':
                    nam_kps = CensusNameColKeypoints(keypoint, img.size)
                    keypoints[cat_id] = nam_kps
                elif label == 'occupation_col':
                    occ_kps = CensusOccupationColKeypoints(keypoint, img.size)
                    keypoints[cat_id] = occ_kps
                elif label == 'veteran_col':
                    vet_kps = CensusVeteranColKeypoints(keypoint, img.size)
                    keypoints[cat_id] = vet_kps

            '''
            keypoints =  CensusNameColKeypoints(keypoint, img.size)
            target.add_field("keypoints", keypoints)

        target = target.clip_to_image(remove_empty=True)

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
