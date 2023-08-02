import os
import subprocess
from custom_logging import log
from config import ROOT_DIR

WD14_TAGGER_PATH = os.path.join(ROOT_DIR, 'library', 'tag_images_by_wd14_tagger.py')

def generate_tags(
        train_data_dir,
        caption_extension,
        batch_size,
        general_threshold,
        character_threshold,
        replace_underscores,
        model,
        recursive,
        max_data_loader_n_workers,
        debug,
        undesired_tags,
        frequency_tags,
):
    log.info(f'Captioning files in {train_data_dir}...')
    run_cmd = f'accelerate launch "{WD14_TAGGER_PATH}"'
    run_cmd += f' --batch_size={int(batch_size)}'
    run_cmd += f' --general_threshold={general_threshold}'
    run_cmd += f' --character_threshold={character_threshold}'
    run_cmd += f' --caption_extension="{caption_extension}"'
    run_cmd += f' --model="{model}"'
    run_cmd += f' --max_data_loader_n_workers="{int(max_data_loader_n_workers)}"'

    if recursive:
        run_cmd += f' --recursive'
    if debug:
        run_cmd += f' --debug'
    if replace_underscores:
        run_cmd += f' --remove_underscore'
    if frequency_tags:
        run_cmd += f' --frequency_tags'

    if not undesired_tags == '':
        run_cmd += f' --undesired_tags="{undesired_tags}"'
    run_cmd += f' "{train_data_dir}"'

    log.info(run_cmd)

    # Run the command
    if os.name == 'posix':
        os.system(run_cmd)
    else:
        subprocess.run(run_cmd)

    log.info('...captioning done')
