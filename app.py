import logging
import uuid
from typing import List, Tuple, Dict, Set

import gradio as gr

from sdriddles.questions import list_all_questions, Question

ALL_QS: List[Tuple[str, Question]] = list_all_questions()

_QUESTION_SESSIONS: Dict[str, Tuple[Set[int], int]] = {}
count = 0

if __name__ == '__main__':
    with gr.Blocks() as demo:
        with gr.Row():
            gr_requirement = gr.Markdown(
                value='(Area for requirements)'
            )

        with gr.Row():
            with gr.Column():
                with gr.Row():
                    gr_prompt = gr.TextArea(placeholder='Prompt', label='Prompt', lines=4)
                with gr.Row():
                    gr_submit = gr.Button(value='Submit', variant='primary', interactive=False)
                with gr.Row():
                    gr_select = gr.Radio(
                        choices=[(title, i) for i, (title, qs) in enumerate(ALL_QS)],
                        label='Select Question'
                    )

            with gr.Column():
                gr_uuid = gr.Text(value='', visible=False)
                with gr.Row():
                    gr_predict = gr.Label(label='Correctness')
                with gr.Row():
                    gr_images = gr.Gallery(value=[], label='Images')
                with gr.Row():
                    gr_explanation = gr.Markdown(value='Area for explanation')


        def _radio_select(uuid_, select_qid):
            global count
            if not uuid_:
                uuid_ = str(uuid.uuid4())
                count += 1
                logging.info(f'Player {count} starts the game now')
            global _QUESTION_SESSIONS
            if uuid_ not in _QUESTION_SESSIONS:
                _QUESTION_SESSIONS[uuid_] = set(), select_qid
            else:
                _exists, _ = _QUESTION_SESSIONS[uuid_]
                _QUESTION_SESSIONS[uuid_] = _exists, select_qid

            title, qs = ALL_QS[select_qid]
            return qs.question_md, {}, [], '', \
                '', gr.Button('Submit', interactive=True), \
                uuid_


        gr_select.select(
            fn=_radio_select,
            inputs=[gr_uuid, gr_select],
            outputs=[gr_requirement, gr_predict, gr_images, gr_explanation, gr_prompt, gr_submit, gr_uuid]
        )


        def _prompt_submit(uuid_, prompt):
            exists, qid = _QUESTION_SESSIONS[uuid_]
            _, qs = ALL_QS[qid]
            correct, image, explanation, extra_images = qs.run(prompt, neg_prompt=None)
            if correct:
                exists.add(qid)
                _QUESTION_SESSIONS[uuid_] = (exists, qid)

            cm = {'Correct': 1.0 if correct else 0.0, 'Wrong': 0.0 if correct else 1.0}
            return cm, [(image, 'Output'), *((img, t) for t, img in extra_images)], explanation


        gr_submit.click(
            fn=_prompt_submit,
            inputs=[gr_uuid, gr_prompt],
            outputs=[gr_predict, gr_images, gr_explanation]
        )

    demo.launch()
