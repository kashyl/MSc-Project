import sd_func
from tagging_func import generate_tags
from config import SDModels, IMG_DIR

undesired_tags = '1girl,1boy,solo,no humans'



def main():
    # sd_func.generate(sd_model=SDModels.sardonyxREDUX_v20, prompt='whale on deep blue sky,white birds,star,thick clouds')
    generate_tags(
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


if __name__ == '__main__':
    main()
