"""Tree trunk detection under varying lighting conditions.

Package layout
--------------
- ``detector``           : YOLO-based detection pipeline (load / infer / filter / count)
- ``preprocessing``      : input validation and image/video loading helpers
- ``lighting_analysis``  : image-based illumination statistics and category labels
- ``lighting_augmentation``: synthetic lighting transformations
- ``video_processor``    : frame-by-frame video / webcam processing
- ``evaluation``         : detection metrics (precision, recall, AP, mAP, count error)
- ``visualization``      : drawing and chart generation
"""

__all__ = [
    "detector",
    "preprocessing",
    "lighting_analysis",
    "lighting_augmentation",
    "video_processor",
    "evaluation",
    "visualization",
]
