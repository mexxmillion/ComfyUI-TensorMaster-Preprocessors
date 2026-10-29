from .nodes import TMIsolateReference, TMCompositeReference
from .prepare import TMDepthReference, TMPrepareReference

NODE_CLASS_MAPPINGS = {"TMIsolateReference": TMIsolateReference, "TMCompositeReference": TMCompositeReference,
                      "TMDepthReference": TMDepthReference, "TMPrepareReference": TMPrepareReference}
NODE_DISPLAY_NAME_MAPPINGS = {"TMIsolateReference": "Tensor Master · Isolate reference",
                            "TMCompositeReference": "Tensor Master · RGB reference background",
                            "TMDepthReference": "Tensor Master · Depth Anything V2 Small",
                            "TMPrepareReference": "Tensor Master · Prepare reference (cached)"}

from .face_eraser import FaceHairEraser

NODE_CLASS_MAPPINGS["FaceHairEraser"] = FaceHairEraser
NODE_DISPLAY_NAME_MAPPINGS["FaceHairEraser"] = "Tensor Master · Face + Hair Eraser (Gauss Fill)"
