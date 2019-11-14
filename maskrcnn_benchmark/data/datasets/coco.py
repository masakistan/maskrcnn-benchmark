# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import defaultdict

import torch
import torchvision

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.keypoint import Keypoints
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
        self.rev_categories = {cat['name']: cat['id'] for cat in self.coco.cats.values()}

        self.keypoints = {}
        self.skeletons = {}
        self.label_offsets = {}
        cur_offset = 0
        for cat in self.coco.cats.values():
            if 'keypoints' in cat:
                self.label_offsets[cat['id']] = (cur_offset, cur_offset + len(cat['keypoints']))
                cur_offset += len(cat['keypoints'])
                self.keypoints[cat['id']] = cat['keypoints']
                self.skeletons[cat['id']] = cat['skeleton']
        self.n_kp_classes = cur_offset
        print('n kp classes', self.n_kp_classes)

        print('kps offsets for labels', self.label_offsets)
        print('cur offset', cur_offset)
        self.keypoint_formats = {
            'name_col': CensusNameColKeypoints,
            'occupation_col': CensusOccupationColKeypoints,
            'veteran_col': CensusVeteranColKeypoints,
        }

        #print('keypoints:', self.keypoints)
        if self.rev_categories['name_col'] in self.keypoints:
            #print("creating name col keypoints")
            CensusNameColKeypoints.CONNECTIONS = self.skeletons[self.rev_categories['name_col']]
        if self.rev_categories['occupation_col'] in self.keypoints:
            CensusOccupationColKeypoints.CONNECTIONS = self.skeletons[self.rev_categories['occupation_col']]
        if self.rev_categories['veteran_col'] in self.keypoints:
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
            keypoints = []
            offsets = [0]
            for i, obj in enumerate(anno):
                keypoint = obj["keypoints"]
                cat_id = obj["category_id"]
                #print('keypiont', keypoint)
                
                padded_keypoint = [0] * self.n_kp_classes * 3
                #print('padded len', len(padded_keypoint))
                start, end = self.label_offsets[cat_id]
                #print('start, end', start, end)
                padded_keypoint[start * 3 :end * 3] = keypoint
                #print(len(padded_keypoint))
                
                offsets.append(offsets[i - 1] + len(padded_keypoint) // 3)
                keypoints.append(padded_keypoint)
            #cats = [obj["category_id"] for obj in anno]
            #offsets = [len(obj["keypoints"]) // 3 for obj in anno]
            #offsets = [0] + offsets
            #print(keypoints)
            #print('offsets', offsets)
            #keypoints = []
            #for x in anno:
            #    keypoint = x["keypoints"]
            #    cat_id = x["category_id"]
            #    keypoints.append(
            #        self.keypoint_formats[self.categories[cat_id]]([keypoint], img.size)
            #    )
                
            target.add_field("keypoints", Keypoints(keypoints, img.size, offsets))

        target = target.clip_to_image(remove_empty=True)

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
