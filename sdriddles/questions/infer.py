from typing import Callable, Optional

from PIL import Image

from hcp.infer import create_image_via_hcpdiff

InferMethodTyping = Callable[[str, Optional[str]], Image.Image]

_INFER_METHODS = {}


def register_infer_method(name: str, method: InferMethodTyping):
    _INFER_METHODS[name] = method


def infer_with_prompt(method_name: str, prompt: str, neg_prompt: Optional[str] = None):
    return _INFER_METHODS[method_name](prompt, neg_prompt)


register_infer_method('hcp_meinamix', create_image_via_hcpdiff)
