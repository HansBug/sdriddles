import os.path
import textwrap
from typing import Optional, Tuple, List

from PIL import Image
from imgutils.detect import detect_faces, detection_visualize

from ..question import Question, register_question

feat_file = os.path.join(os.path.dirname(__file__), 'character_feature.npy')


class Faces2Question(Question):
    def __init__(self):
        Question.__init__(
            self,
            question_md=textwrap.dedent("""
                Draw images with 2 faces
            """).strip(),
            neg_prompt_required=False,
        )

    def check(self, prompt: str, neg_prompt: Optional[str], image: Image.Image) \
            -> Tuple[bool, str, List[Tuple[str, Image.Image]]]:
        detection = detect_faces(image)
        visual = detection_visualize(image, detection)
        if len(detection) == 2:
            return True, '2 Faces Detected', [('Detection', visual)]
        else:
            return True, f'2 Faces Expected, But {len(detection)} Found.', [('Detection', visual)]


register_question((1, 2), '2 Faces', Faces2Question())
