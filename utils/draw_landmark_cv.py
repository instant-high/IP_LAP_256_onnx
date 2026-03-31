import math
from typing import List, Mapping, Optional, Tuple, Union
import dataclasses
import cv2
import numpy as np

# Constants
_BGR_CHANNELS = 3
WHITE_COLOR = (224, 224, 224)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 128, 0)
BLUE_COLOR = (255, 0, 0)

_PRESENCE_THRESHOLD = 0.5
_VISIBILITY_THRESHOLD = 0.5

# Simple Landmark class to replace MediaPipe landmarks
class Landmark:
    def __init__(self, idx: int, x: float, y: float):
        self.idx = idx
        self.x = x
        self.y = y

@dataclasses.dataclass
class DrawingSpec:
    color: Tuple[int, int, int] = WHITE_COLOR
    thickness: int = 2
    circle_radius: int = 2

# Helper: normalized coords -> pixel coords
def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int, image_height: int
) -> Union[None, Tuple[int, int]]:
    if not (0 <= normalized_x <= 1) or not (0 <= normalized_y <= 1):
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px

# Example landmark connections (replace or extend as needed)
FACEMESH_LIPS = frozenset([
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314),
    (314, 405), (405, 321), (321, 375), (375, 291)
])

FACEMESH_FULL = FACEMESH_LIPS  # Add other sets like eyes, nose, etc. as needed

def summary_landmark(edge_set):
    landmarks = set()
    for a, b in edge_set:
        landmarks.add(a)
        landmarks.add(b)
    return landmarks

all_landmark_idx = summary_landmark(FACEMESH_FULL)


def draw_landmarks(
    image: np.ndarray,
    landmark_list: List[Landmark],
    connections: Optional[List[Tuple[int, int]]] = None,
    landmark_drawing_spec: Union[DrawingSpec, Mapping[int, DrawingSpec]] = DrawingSpec(color=RED_COLOR),
    connection_drawing_spec: Union[DrawingSpec, Mapping[Tuple[int, int], DrawingSpec]] = DrawingSpec(),
    draw_points: bool = False,
    overlay_alpha: float = 0.0  # 0 = draw directly, 0<alpha<=1 = blend overlay
) -> np.ndarray:
    if not landmark_list:
        return image
    if image.shape[2] != _BGR_CHANNELS:
        raise ValueError('Input image must be BGR with 3 channels.')

    if overlay_alpha > 0:
        overlay = image.copy()
    else:
        overlay = image

    image_rows, image_cols, _ = image.shape
    idx_to_coordinates = {}
    for landmark in landmark_list:
        landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y, image_cols, image_rows)
        if landmark_px:
            idx_to_coordinates[landmark.idx] = landmark_px

    # Draw connections
    for connection in connections or []:
        start_idx, end_idx = connection
        if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
            spec = connection_drawing_spec[connection] if isinstance(connection_drawing_spec, Mapping) else connection_drawing_spec
            cv2.line(overlay, idx_to_coordinates[start_idx], idx_to_coordinates[end_idx], spec.color, spec.thickness)

    # Draw landmark points
    if draw_points:
        for idx, coord in idx_to_coordinates.items():
            spec = landmark_drawing_spec[idx] if isinstance(landmark_drawing_spec, Mapping) else landmark_drawing_spec
            cv2.circle(overlay, coord, spec.circle_radius, spec.color, spec.thickness)

    if overlay_alpha > 0:
        return cv2.addWeighted(image, 1.0 - overlay_alpha, overlay, overlay_alpha, 0)
        
    return overlay