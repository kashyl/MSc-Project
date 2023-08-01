import requests, base64, time, os, io
from PIL import Image, PngImagePlugin
from config import SDModelType, PromptPrefix, SD_URL, IMG_DIR as OUTPUT_DIR
from custom_logging import setup_logging, log_function_call, log_image_generation_progress
from typing import Dict, Any, List
import aiohttp, asyncio

log = setup_logging(debug=True)


def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1
    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1
    return path

def save_image(json_resp):
    for i in json_resp['images']:
        image = Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0])))

        png_payload = {"image": "data:image/png;base64," + i}
        response2 = requests.post(url=f'{SD_URL}/sdapi/v1/png-info', json=png_payload)

        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("parameters", response2.json().get("info"))
        
        timestr = time.strftime("%Y%m%d-%H%M%S")
        file_path = os.path.join(OUTPUT_DIR, f'{timestr}.png')
        file_path = uniquify(file_path)

        image.save(file_path, pnginfo=pnginfo)

def add_prefix_to_prompt(prompt: str, prefix: str) -> str:
    return ", \n\n".join([_ for _ in [prefix, prompt] if _])

def generate_payloads(
        model_payload: Dict[str, Any], 
        model_prefixes: PromptPrefix, 
        prompt: str, 
        prompt_n: str, 
        gen_count: int) -> List[Dict[str, Any]]:
    payloads = []
    positive_prompt = add_prefix_to_prompt(prompt, model_prefixes.positive)
    negative_prompt = add_prefix_to_prompt(prompt_n, model_prefixes.negative)

    payload = model_payload.copy()  # Make a shallow copy of the model_payload
    payload["prompt"] = positive_prompt
    payload["negative_prompt"] = negative_prompt

    payloads = [payload] * gen_count

    return payloads

def update_model_options(model_option_payload: Dict[str, Any]):
    opt = requests.get(url=f'{SD_URL}/sdapi/v1/options')
    opt_json = opt.json()
    new_options = {**opt_json, **model_option_payload}
    requests.post(url=f'{SD_URL}/sdapi/v1/options', json=new_options)

async def request_image(session, data):
    async with session.post(f'{SD_URL}/sdapi/v1/txt2img', json=data) as response:
        return await response.json()
    
async def check_progress(session):
    """
    {
        "progress": 0,
        "eta_relative": 0,
        "state": {},
        "current_image": "string",
        "textinfo": "string"
    }
    """
    async with session.get(f'{SD_URL}/sdapi/v1/progress') as response:
        progress_response = await response.json()
        progress = progress_response['progress']
        log.info(f"Progress: {progress}")

async def generate(sd_model: SDModelType, prompt: str, prompt_n: str = None, gen_count: int = 1):
    update_model_options(sd_model.option_payload)

    payloads = generate_payloads(
        model_payload=sd_model.payload, 
        model_prefixes=sd_model.prompt_prefix, 
        prompt=prompt, 
        prompt_n=prompt_n, 
        gen_count=gen_count
    )

    log.info(f'Starting generation of {len(payloads)} image(s) ...')

    for i, payload in enumerate(payloads):
        start_time = time.time()

        async with aiohttp.ClientSession() as session:
            # Create the POST task for the image request
            task_post = asyncio.create_task(request_image(session, payload))

            # Check the progress and sleep every second while waiting for the POST task to complete
            while not task_post.done():
                await check_progress(session)
                await asyncio.sleep(1)

            # Get the result of the POST task
            resp = await task_post

        save_image(json_resp=resp)
        log_image_generation_progress(i, len(payloads), start_time)

    log.info(f"Finished generating images. Saved at: {OUTPUT_DIR}")
