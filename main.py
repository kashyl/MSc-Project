import sd_api_wrapper as sd_api_wrapper
from wd14_tagger_runner import generate_tags
from config import SDModels, IMG_DIR
import os, asyncio
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow debug & info (build for CPU optimization etc.)

undesired_tags = '1girl,1boy,solo,no humans'

# ratings: general, sensitive, questionable, explicit
# todo: filtering (only general allowed)


def main():
    asyncio.run(sd_api_wrapper.generate(sd_model=SDModels.sardonyxREDUX_v20, prompt='whale on deep blue sky,white birds,star,thick clouds'))
    
    # generate_tags(
    #     train_data_dir=IMG_DIR,
    #     caption_extension='.json',
    #     batch_size=8,
    #     general_threshold=0.35,
    #     character_threshold=0.35,
    #     replace_underscores=True,
    #     model='SmilingWolf/wd-v1-4-convnextv2-tagger-v2',
    #     recursive=False,
    #     max_data_loader_n_workers=2,
    #     debug=True,
    #     undesired_tags=undesired_tags,
    #     frequency_tags=False,
    # )


if __name__ == '__main__':
    main()
