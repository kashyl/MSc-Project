import sd_api_wrapper as sd_api_wrapper
from image_tagging import image_tagger
from config import SDModels, IMG_DIR
import os, asyncio

def main():

    image_tagger.tag_image(sd_api_wrapper.get_first_image())

    # asyncio.run(sd_api_wrapper.generate_image(
    #     sd_model=SDModels.sardonyxREDUX_v20, 
    #     prompt='whale on deep blue sky,white birds,star,thick clouds'
    #     )
    # )

if __name__ == '__main__':
    main()
