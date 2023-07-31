import requests, io, base64, os, time
from PIL import Image, PngImagePlugin
from config import SDModelType, PromptPrefix, SD_URL, IMG_DIR as OUTPUT_DIR
from custom_logging import setup_logging, log_function_call, log_image_generation_progress
from typing import Dict, Any, List

log = setup_logging(debug=True)


def _uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1
    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1
    return path

def _save_image(json_resp):
    for i in json_resp['images']:
        image = Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0])))

        png_payload = {"image": "data:image/png;base64," + i}
        response2 = requests.post(url=f'{SD_URL}/sdapi/v1/png-info', json=png_payload)

        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("parameters", response2.json().get("info"))
        
        timestr = time.strftime("%Y%m%d-%H%M%S")
        file_path = os.path.join(OUTPUT_DIR, f'{timestr}.png')
        file_path = _uniquify(file_path)

        image.save(file_path, pnginfo=pnginfo)

def _add_prefix_to_prompt(prompt: str, prefix: str) -> str:
    return ", \n\n".join([_ for _ in [prefix, prompt] if _])

def _generate_payloads(
        model_payload: Dict[str, Any], 
        model_prefixes: PromptPrefix, 
        prompt: str, 
        prompt_n: str, 
        gen_count: int) -> List[Dict[str, Any]]:
    payloads = []
    positive_prompt = _add_prefix_to_prompt(prompt, model_prefixes.positive)
    negative_prompt = _add_prefix_to_prompt(prompt_n, model_prefixes.negative)

    payload = model_payload.copy()  # Make a shallow copy of the model_payload
    payload["prompt"] = positive_prompt
    payload["negative_prompt"] = negative_prompt

    payloads = [payload] * gen_count

    return payloads

def _update_model_options(model_option_payload: Dict[str, Any]):
    opt = requests.get(url=f'{SD_URL}/sdapi/v1/options')
    opt_json = opt.json()
    opt_json['sd_model_checkpoint'] = '<MODEL_NAME>'
    new_options = {**opt_json, **model_option_payload}
    requests.post(url=f'{SD_URL}/sdapi/v1/options', json=new_options)

def generate(sd_model: SDModelType, prompt: str, prompt_n: str = None, gen_count: int = 1):
    _update_model_options(sd_model.option_payload)

    payloads = _generate_payloads(
        model_payload=sd_model.payload, 
        model_prefixes=sd_model.prompt_prefix, 
        prompt=prompt, 
        prompt_n=prompt_n, 
        gen_count=gen_count
    )

    log.info(f'Starting generation of {len(payloads)} image(s) ...')

    for i, payload in enumerate(payloads):
        start_time = time.time()

        response = requests.post(url=f'{SD_URL}/sdapi/v1/txt2img', json=payload)
        resp = response.json()
        _save_image(json_resp=resp)

        log_image_generation_progress(i, len(payloads), start_time)

    log.info(f"Finished generating images. Saved at: {OUTPUT_DIR}")
