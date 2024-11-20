# utils/partbased_segment.py

import torch

def get_segmentation():
    # Define segmentation groups as in your setup
    segmentation = {
        1: [44, 45, 46, 47, 48, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98], # head and neck
        2: [0, 1, 4, 16, 17, 20, 32, 37, 80, 83, 84],  # trunk
        3: [19, 21, 22, 23, 26, 31, 33, 34, 80, 83, 84],  # right upper limb
        4: [3, 5, 6, 7, 10, 15, 35, 36, 79, 81, 82],  # left upper limb
        5: [64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142],  # right hand
        6: [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120],  # left hand
        7: [2, 8, 9, 11, 12, 13, 14, 38, 39, 40],  # left legs
        8: [18, 24, 25, 27, 28, 29, 30, 41, 42, 43]  # right legs
    }
    return segmentation

def create_marker_to_part(segmentation, n_markers):
    marker_to_part = torch.zeros(n_markers, dtype=torch.long)
    for part_label, indices in segmentation.items():
        for idx in indices:
            marker_to_part[idx] = part_label
    return marker_to_part
