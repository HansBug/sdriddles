import glob
import os.path
from typing import Optional

from hbutils.system import TemporaryDirectory
from hcpdiff.utils import load_config_with_cli
from hcpdiff.visualizer_reloadable import VisualizerReloadable
from imgutils.data import load_image

from sdriddles.utils import data_to_cli_args

_DEFAULT_NEG_PROMPT = ('lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, '
                       'cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, '
                       'username, blurry')
_DEFAULT_CFG_FILE = os.path.join(os.path.dirname(__file__), 'infer.yaml')

_VIS = None


def create_image_via_hcpdiff(prompt: str, neg_prompt: Optional[str] = None, seed: Optional[int] = None,
                             cfg_file: Optional[str] = None, n_repeats: int = 1,
                             extra_cfgs: Optional[dict] = None):
    neg_prompt = neg_prompt or _DEFAULT_NEG_PROMPT
    cfg_file = cfg_file or _DEFAULT_CFG_FILE
    with TemporaryDirectory() as td:
        while True:
            try:
                cfgs = load_config_with_cli(
                    cfg_file,
                    args_list=data_to_cli_args({
                        **(extra_cfgs or {}),
                        'output_dir': td,
                        'N_repeats': n_repeats,
                    })
                )

                global _VIS
                if _VIS is None:
                    _VIS = VisualizerReloadable(cfgs)
                else:
                    _VIS.check_reload(cfgs)

                _VIS.vis_to_dir(
                    prompt=[prompt],
                    negative_prompt=[neg_prompt],
                    seeds=[seed],
                    save_cfg=True,
                    **cfgs.infer_args
                )
            except RuntimeError:
                n_repeats += 1
            else:
                break

        png_file = glob.glob(os.path.join(td, '*.png'))[0]
        return load_image(png_file)
