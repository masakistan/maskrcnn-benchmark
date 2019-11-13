from maskrcnn_benchmark.structures.keypoint import Keypoints, _create_flip_indices
from .census_keypoint_names import keypoint_names


class CensusNameColKeypoints(Keypoints):
    NAMES = keypoint_names["name_col"]
    FLIP_MAP = {}
CensusNameColKeypoints.FLIP_INDS = _create_flip_indices(CensusNameColKeypoints.NAMES, CensusNameColKeypoints.FLIP_MAP)

class CensusOccupationColKeypoints(Keypoints):
    NAMES = keypoint_names["occupation_col"]
    FLIP_MAP = {}
CensusOccupationColKeypoints.FLIP_INDS = _create_flip_indices(CensusOccupationColKeypoints.NAMES, CensusOccupationColKeypoints.FLIP_MAP)


class CensusVeteranColKeypoints(Keypoints):
    NAMES = keypoint_names["veteran_col"]
    FLIP_MAP = {}
CensusVeteranColKeypoints.FLIP_INDS = _create_flip_indices(CensusVeteranColKeypoints.NAMES, CensusVeteranColKeypoints.FLIP_MAP)
    
