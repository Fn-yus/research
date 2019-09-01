import numpy as np

def main(needle, scale):
    scale_upper_list, = np.where(needle <= scale)
    scale_upper_index = min(scale_upper_list)
    scale_lower_list, = np.where(needle >= scale)
    scale_lower_index = max(scale_lower_list)
    needle_percentage = needle/(scale[scale_upper_index] - scale[scale_lower_index])
    needle_position = (scale_lower_index - 3) + needle_percentage
    return needle_position