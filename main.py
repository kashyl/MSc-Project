from wd14_caption_gui import caption_images
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(ROOT_DIR, 'img')

undesired_tags = '1girl,1boy,solo,no humans'


if __name__ == '__main__':
    caption_images(
        train_data_dir=IMG_DIR,
        caption_extension='.json',
        batch_size=8,
        general_threshold=0.35,
        character_threshold=0.35,
        replace_underscores=True,
        model='SmilingWolf/wd-v1-4-convnextv2-tagger-v2',
        recursive=False,
        max_data_loader_n_workers=2,
        debug=True,
        undesired_tags=undesired_tags,
        frequency_tags=False,
    )
