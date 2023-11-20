import os
import warnings
from typing import Tuple, Optional, Dict, List

from PIL import Image

from .infer import infer_with_prompt


class Question:
    def __init__(self, question_md: str, neg_prompt_required: bool = False):
        self.question_md = question_md
        self.neg_prompt_required = neg_prompt_required

    def infer(self, prompt: str, neg_prompt: Optional[str] = None) -> Image.Image:
        if not self.neg_prompt_required:
            if neg_prompt:
                warnings.warn(f'Negative prompt not required, value for neg_prompt will be ignored: {neg_prompt!r}.')
                neg_prompt = None
        else:
            if neg_prompt is None:
                raise ValueError('Negative prompted required.')

        return infer_with_prompt(os.environ['INFER_METHOD'], prompt, neg_prompt)

    def check(self, prompt: str, neg_prompt: Optional[str], image: Image.Image) \
            -> Tuple[bool, str, List[Tuple[str, Image.Image]]]:
        raise NotImplementedError

    def run(self, prompt: str, neg_prompt: Optional[str] = None) \
            -> Tuple[bool, Image.Image, str, List[Tuple[str, Image.Image]]]:
        image = self.infer(prompt, neg_prompt)
        correct, explanation_md, extra_images = self.check(prompt, neg_prompt, image)
        return correct, image, explanation_md, extra_images


_ALL_QUESTIONS: Dict[Tuple[int, ...], Tuple[str, Question]] = {}


def register_question(qno: Tuple[int, ...], name: str, question: Question):
    _ALL_QUESTIONS[qno] = (name, question)


def list_all_questions() -> List[Tuple[str, Question]]:
    return [
        (
            '-'.join(map(str, key)) + ' ' + _ALL_QUESTIONS[key][0],
            _ALL_QUESTIONS[key][1]
        )
        for key in sorted(_ALL_QUESTIONS.keys())
    ]
