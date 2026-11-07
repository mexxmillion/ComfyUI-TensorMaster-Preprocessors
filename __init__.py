from .nodes import TMIsolateReference, TMCompositeReference
from .prepare import TMDepthReference, TMPrepareReference
from .video import TMPrepareVideo

NODE_CLASS_MAPPINGS = {"TMPrepareVideo": TMPrepareVideo,
                      "TMIsolateReference": TMIsolateReference, "TMCompositeReference": TMCompositeReference,
                      "TMDepthReference": TMDepthReference, "TMPrepareReference": TMPrepareReference}
NODE_DISPLAY_NAME_MAPPINGS = {"TMPrepareVideo": "Tensor Master · Prepare video · target size / 24 fps",
                            "TMIsolateReference": "Tensor Master · Isolate reference",
                            "TMCompositeReference": "Tensor Master · RGB reference background",
                            "TMDepthReference": "Tensor Master · Depth Anything V2 Small",
                            "TMPrepareReference": "Tensor Master · Prepare reference (cached)"}

from .face_eraser import FaceHairEraser

NODE_CLASS_MAPPINGS["FaceHairEraser"] = FaceHairEraser
NODE_DISPLAY_NAME_MAPPINGS["FaceHairEraser"] = "Tensor Master · Face + Hair Eraser (Gauss Fill)"

from .reference_manager.node import TMH3ReferenceManager

NODE_CLASS_MAPPINGS['TMH3ReferenceManager'] = TMH3ReferenceManager
NODE_DISPLAY_NAME_MAPPINGS['TMH3ReferenceManager'] = 'TM · H3 Reference Manager'
WEB_DIRECTORY = './web'

from .face_cover import TMFaceCover

NODE_CLASS_MAPPINGS['TMFaceCover'] = TMFaceCover
NODE_DISPLAY_NAME_MAPPINGS['TMFaceCover'] = 'TM · Face Cover (solid ellipse)'

from .lora_loader import TMMultiLora

NODE_CLASS_MAPPINGS['TMMultiLora'] = TMMultiLora
NODE_DISPLAY_NAME_MAPPINGS['TMMultiLora'] = 'TM · Power LoRA'

from .state_switch import TMStateMenu, TMStateSwitch

NODE_CLASS_MAPPINGS.update(TMStateMenu=TMStateMenu, TMStateSwitch=TMStateSwitch)
NODE_DISPLAY_NAME_MAPPINGS.update(TMStateMenu='TM · State Menu', TMStateSwitch='TM · State Switch')
