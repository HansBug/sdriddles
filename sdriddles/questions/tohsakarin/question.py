import os.path
import textwrap
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from imgutils.metrics import ccip_batch_same

from ..question import Question, register_question

feat_file = os.path.join(os.path.dirname(__file__), 'character_feature.npy')


class TohsakaRinQuestion(Question):
    def __init__(self):
        Question.__init__(
            self,
            question_md=textwrap.dedent("""
                Draw image for tohsaka rin
            """).strip(),
            neg_prompt_required=False,
        )

        self.feats = [item for item in np.load(feat_file)]

    def check(self, prompt: str, neg_prompt: Optional[str], image: Image.Image) -> Tuple[bool, str]:
        score = ccip_batch_same([image, *self.feats])[0, 1:].mean()
        if score >= 0.8:
            return True, f'Yes she is tohsaka rin (score: {score:.2f})'
        else:
            return False, f'I don\'t think she is tohsaka rin (socre: {score:.2f})'


register_question((1, 1), 'Tohsaka Rin', TohsakaRinQuestion())
