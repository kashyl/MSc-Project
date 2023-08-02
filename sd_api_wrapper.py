import requests, base64, time, os, io
from PIL import Image, PngImagePlugin
from config import SDModelType, PromptPrefix, SD_URL, IMG_DIR as OUTPUT_DIR
from custom_logging import setup_logging, log_function_call
from typing import Dict, Any, List, Tuple, Optional
import aiohttp, asyncio

log = setup_logging(debug=True)

def uniquify(path: str) -> str:
    filename, extension = os.path.splitext(path)
    counter = 1
    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1
    return path

def extract_base64(encoding: str) -> str:
    if encoding.startswith("data:image/"):
        encoding = encoding.split(";")[1].split(",")[1]
    return encoding

def decode_base64_to_image(encoding: str) -> Tuple[Optional[Image.Image], Optional[bytes]]:
    try:
        image_data = base64.b64decode(encoding)
        image = Image.open(io.BytesIO(image_data))
        return image, image_data
    except Exception as e:
        log.warning(f'Cannot decode image: {e}')
        return None, None

def get_metadata(image_data: bytes) -> str:
    png_payload = {"image": "data:image/png;base64," + base64.b64encode(image_data).decode()}
    response = requests.post(url=f'{SD_URL}/sdapi/v1/png-info', json=png_payload)
    return response.json().get("info")

def add_metadata(image: Image.Image, metadata: str) -> Image.Image:
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("parameters", metadata)

    # Create a bytes buffer to hold the new image data
    output = io.BytesIO()
    image.save(output, format="PNG", pnginfo=pnginfo)
    
    # Convert bytes back to a PIL Image object
    image_with_metadata = Image.open(output)
    return image_with_metadata

def save_image(image: Image.Image) -> None:    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    file_path = os.path.join(OUTPUT_DIR, f'{timestr}.png')
    file_path = uniquify(file_path)
    image.save(file_path)
    log.info(f'Image saved: {file_path}')


def add_prefix_to_prompt(prompt: str, prefix: str) -> str:
    return ", \n\n".join([_ for _ in [prefix, prompt] if _])

def create_payload(
        model_payload: Dict[str, Any], 
        model_prefixes: PromptPrefix, 
        prompt: str, 
        prompt_n: str) -> List[Dict[str, Any]]:
    positive_prompt = add_prefix_to_prompt(prompt, model_prefixes.positive)
    negative_prompt = add_prefix_to_prompt(prompt_n, model_prefixes.negative)

    payload = model_payload.copy()  # Make a shallow copy of the model_payload
    payload["prompt"] = positive_prompt
    payload["negative_prompt"] = negative_prompt

    return payload

def update_model_options(model_option_payload: Dict[str, Any]):
    opt = requests.get(url=f'{SD_URL}/sdapi/v1/options')
    opt_json = opt.json()
    new_options = {**opt_json, **model_option_payload}
    requests.post(url=f'{SD_URL}/sdapi/v1/options', json=new_options)

async def request_image(session, data):
    async with session.post(f'{SD_URL}/sdapi/v1/txt2img', json=data) as response:
        return await response.json()

def as_percentage(progress_float):
    return f"{progress_float * 100:.2f}%"

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
        eta = progress_response['eta_relative']
        log.info(f"Progress: {as_percentage(progress)}, ETA: {eta:.2f}s")

@log_function_call
async def generate_image(sd_model: SDModelType, prompt: str, prompt_n: str = None) -> Image.Image:
    update_model_options(sd_model.option_payload)

    payload = create_payload(
        model_payload=sd_model.payload, 
        model_prefixes=sd_model.prompt_prefix, 
        prompt=prompt, 
        prompt_n=prompt_n
    )

    async with aiohttp.ClientSession() as session:
        # Create the POST task for the image request
        task_post = asyncio.create_task(request_image(session, payload))

        # Check the progress and sleep every second while waiting for the POST task to complete
        while not task_post.done():
            await check_progress(session)
            await asyncio.sleep(1)

        # Get the result of the POST task
        resp = await task_post
        # resp = {'images': ['bjkahsdjsa...', 'askjdhas...', 'ajkjhdashda...']}

    image_base64 = extract_base64(resp.get('images', [])[0]) # Fixed indexing
    image, image_data = decode_base64_to_image(image_base64)

    metadata = get_metadata(image_data)
    image_with_metadata = add_metadata(image, metadata)
    return image_with_metadata
