import torch
from itertools import compress
from maskrcnn_benchmark.modeling.utils import cat


# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


class Keypoints(object):
    def __init__(self, keypoints, size, offsets, mode=None):
        # FIXME remove check once we have better integration with device
        # in my version this would consistently return a CPU tensor
        #print(keypoints)
        device = keypoints.device if isinstance(keypoints, torch.Tensor) else torch.device('cpu')
        if isinstance(keypoints, list):
            keypoints = [torch.as_tensor(x, dtype=torch.float32, device=device) for x in keypoints]
            keypoints = cat(keypoints, dim=0)

        #print(keypoints)
        keypoints = torch.as_tensor(keypoints, dtype=torch.float32, device=device)
        
        self.offsets = offsets
        num_keypoints = keypoints.shape[0]
        #print("num keypoints", num_keypoints)
        #print("keypoints shape", keypoints.shape)
        if num_keypoints:
            keypoints = keypoints.view(-1, 11)
        #print('loaded keypoints', keypoints.shape)
        #assert keypoints.shape[0] + 1 == len(offsets), "Keypoints and offsets oar of different length {} {}".format(keypoints.shape[0], len(offsets))
        
        # TODO should I split them?
        # self.visibility = keypoints[..., 2]
        self.keypoints = keypoints# [..., :2]
        #print("self keypoints", self.keypoints.shape)

        self.size = size
        self.mode = mode
        self.extra_fields = {}

    def crop(self, box):
        raise NotImplementedError()

    def resize(self, size, *args, **kwargs):
        #print('resizing')
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))
        ratio_w, ratio_h = ratios
        resized_data = self.keypoints.clone()
        #print('size', size, self.size)
        #print('rations', ratio_w, ratio_h)
        #print('w before', resized_data[..., 0])
        #print('w before', resized_data[..., 1])
        resized_data[..., 0] *= ratio_w
        resized_data[..., 3] *= ratio_w
        resized_data[..., 5] *= ratio_w
        resized_data[..., 7] *= ratio_w
        resized_data[..., 9] *= ratio_w
        
        resized_data[..., 1] *= ratio_h
        resized_data[..., 4] *= ratio_w
        resized_data[..., 6] *= ratio_w
        resized_data[..., 8] *= ratio_w
        resized_data[..., 10] *= ratio_w

        #print('w after', resized_data[..., 0])
        #print('w after', resized_data[..., 1])
        keypoints = type(self)(resized_data, size, self.offsets, self.mode)
        for k, v in self.extra_fields.items():
            keypoints.add_field(k, v)
        return keypoints

    def transpose(self, method):
        if method not in (FLIP_LEFT_RIGHT,):
            raise NotImplementedError(
                    "Only FLIP_LEFT_RIGHT implemented")

        flip_inds = type(self).FLIP_INDS
        flipped_data = self.keypoints[:, flip_inds]
        width = self.size[0]
        TO_REMOVE = 1
        # Flip x coordinates
        flipped_data[..., 0] = width - flipped_data[..., 0] - TO_REMOVE

        # Maintain COCO convention that if visibility == 0, then x, y = 0
        inds = flipped_data[..., 2] == 0
        flipped_data[inds] = 0

        keypoints = type(self)(flipped_data, self.size, self.offsets, self.mode)
        for k, v in self.extra_fields.items():
            keypoints.add_field(k, v)
        return keypoints

    def to(self, *args, **kwargs):
        keypoints = type(self)(self.keypoints.to(*args, **kwargs), self.size, self.offsets, self.mode)
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            keypoints.add_field(k, v)
        return keypoints

    def keypoint_indices_and_offsets(self, idxs):
        #print('idxs', idxs)
        if isinstance(idxs, int):
            idxs = [idxs]
        elif isinstance(idxs, tuple):
            print("WARNING: tuple may not work as intended {}".format(idxs))
            idxs = [idxs[1]]
            
        if isinstance(idxs, torch.BoolTensor) or isinstance(idxs, torch.cuda.BoolTensor):
            #print("getting bool")
            ret = cat([torch.arange(self.offsets[i], self.offsets[i + 1]) for i, x in enumerate(idxs) if x], dim=0)
            #new_offsets = [0] + [self.offsets[i + 1] - self.offsets[i] for i, x in enumerate(idxs) if x]
            #print('bool:', ret, new_offsets)
            new_offsets = [0]
            for i, x in enumerate(idxs):
                if x:
                    new_offsets.append(new_offsets[-1] + self.offsets[i + 1] - self.offsets[i])
        elif isinstance(idxs, list) or isinstance(idxs, torch.cuda.LongTensor) or isinstance(idxs, torch.LongTensor):
            #print("getting index")
            ret = cat([torch.arange(self.offsets[i], self.offsets[i + 1]) for i in idxs if i >= 0], dim=0)
            new_offsets = [0]
            for i in idxs:
                if i >= 0:
                    new_offsets.append(new_offsets[-1] + self.offsets[i + 1] - self.offsets[i])
        #print('new offsets', new_offsets)

        return ret, new_offsets
 
    def __getitem__(self, item):
        #print('item', item)
        keypoint_idxs, new_offsets = self.keypoint_indices_and_offsets(item)
        keypoints = type(self)(self.keypoints[keypoint_idxs], self.size, new_offsets, self.mode)
        for k, v in self.extra_fields.items():
            keypoints.add_field(k, v[item])
        return keypoints

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += 'num_instances={}, '.format(len(self.offsets) - 1)
        s += 'image_width={}, '.format(self.size[0])
        s += 'image_height={}, '.format(self.size[1])
        s += 'num_keypoints={})'.format(len(self.keypoints))
        return s


def _create_flip_indices(names, flip_map):
    full_flip_map = flip_map.copy()
    full_flip_map.update({v: k for k, v in flip_map.items()})
    flipped_names = [i if i not in full_flip_map else full_flip_map[i] for i in names]
    flip_indices = [names.index(i) for i in flipped_names]
    return torch.tensor(flip_indices)


# TODO make this nicer, this is a direct translation from C2 (but removing the inner loop)
def keypoints_to_heat_map(keypoints, rois, heatmap_size):
    #print('keypionts heat map', keypoints[0], keypoints.shape)
    #print('heatmap_size', heatmap_size)
    #print('rois', rois, rois.shape)
    if rois.numel() == 0:
        return rois.new().long(), rois.new().long()


    heatmaps = []
    valids = []
    #print('keypoints', keypoints)
    for i, (roi, keypoint) in enumerate(zip(rois, keypoints)):
        offset_x = roi[0]
        offset_y = roi[1]
        scale_x = heatmap_size / (roi[2] - roi[0])
        scale_y = heatmap_size / (roi[3] - roi[1])

        offset_x = offset_x[None]
        offset_y = offset_y[None]
        scale_x = scale_x[None]
        scale_y = scale_y[None]

        #print('kps', keypoints.shape)
        #print('roi', roi)
        #print('keypoint'#, keypoint)
        keypoint = keypoint.keypoints
        #print('keypoint', keypoint.shape)
        x = keypoint[..., 0]
        y = keypoint[..., 1]
        #print('x', x)
        #print('y', y)
        #print('rois 2', rois[:, 2])
        #print('rois 3', rois[:, 3])
        
        x_boundary_inds = x == roi[2][None]
        y_boundary_inds = y == roi[3][None]
        #print('x boundary', x_boundary_inds)
        #print('y boundary', y_boundary_inds)
        
        x = (x - offset_x) * scale_x
        x = x.floor().long()
        y = (y - offset_y) * scale_y
        y = y.floor().long()
        
        #print('scaled x', x)
        #print('scaled y', y)
        
        x[x_boundary_inds] = heatmap_size - 1
        y[y_boundary_inds] = heatmap_size - 1
        
        valid_loc = (x >= 0) & (y >= 0) & (x < heatmap_size) & (y < heatmap_size)
        vis = keypoint[..., 2] > 0
        valid = (valid_loc & vis).long()
        
        lin_ind = y * heatmap_size + x
        heatmap = lin_ind * valid

        heatmaps.append(heatmap)
        valids.append(valid)
    heatmaps = cat(heatmaps)
    valids = cat(valids)
    #print('heatmaps', heatmaps, heatmaps.shape)
    #print('valid', valids, valids.shape)

    return heatmaps, valids

class PersonKeypoints(Keypoints):
    NAMES = [
        'nose',
        'left_eye',
        'right_eye',
        'left_ear',
        'right_ear',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip',
        'right_hip',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle'
    ]
    FLIP_MAP = {
        'left_eye': 'right_eye',
        'left_ear': 'right_ear',
        'left_shoulder': 'right_shoulder',
        'left_elbow': 'right_elbow',
        'left_wrist': 'right_wrist',
        'left_hip': 'right_hip',
        'left_knee': 'right_knee',
        'left_ankle': 'right_ankle'
    }


# TODO this doesn't look great
PersonKeypoints.FLIP_INDS = _create_flip_indices(PersonKeypoints.NAMES, PersonKeypoints.FLIP_MAP)
def kp_connections(keypoints):
    kp_lines = [
        [keypoints.index('left_eye'), keypoints.index('right_eye')],
        [keypoints.index('left_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('right_ear')],
        [keypoints.index('left_eye'), keypoints.index('left_ear')],
        [keypoints.index('right_shoulder'), keypoints.index('right_elbow')],
        [keypoints.index('right_elbow'), keypoints.index('right_wrist')],
        [keypoints.index('left_shoulder'), keypoints.index('left_elbow')],
        [keypoints.index('left_elbow'), keypoints.index('left_wrist')],
        [keypoints.index('right_hip'), keypoints.index('right_knee')],
        [keypoints.index('right_knee'), keypoints.index('right_ankle')],
        [keypoints.index('left_hip'), keypoints.index('left_knee')],
        [keypoints.index('left_knee'), keypoints.index('left_ankle')],
        [keypoints.index('right_shoulder'), keypoints.index('left_shoulder')],
        [keypoints.index('right_hip'), keypoints.index('left_hip')],
    ]
    return kp_lines
PersonKeypoints.CONNECTIONS = kp_connections(PersonKeypoints.NAMES)

