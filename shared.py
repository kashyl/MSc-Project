import os, re
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from enum import Enum

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(ROOT_DIR, 'img')
STATIC_DIR = os.path.join(ROOT_DIR, 'static')
MOCK_GEN_IMG_PATH = os.path.join(STATIC_DIR, 'debug.png')
SD_URL = "http://127.0.0.1:7860"
RANDOM_MODEL_OPT_STRING = 'Select at random for each generation'

class DifficultyLevels(Enum):
    EASY = "Easy"
    NORMAL = "Normal"
    HARD = "Hard"

DIFFICULTY_LEVEL_TAG_RATIO = {
    DifficultyLevels.EASY: 0.5,
    DifficultyLevels.NORMAL: 1,
    DifficultyLevels.HARD: 2
}

DIFFICULTY_LEVEL_EXP_GAIN = {
    DifficultyLevels.EASY: 0.5,
    DifficultyLevels.NORMAL: 1.0,
    DifficultyLevels.HARD: 1.5
}

class GUIFiltersLabels(Enum):
    GENERAL = 'General (Safe)'
    SENSITIVE = 'Sensitive'
    QUESTIONABLE = 'Questionable'
    EXPLICIT = 'Explicit (Show Everything)'

class EventHandler():
    """ For updating progress in the GUI. """
    def __init__(self):
        self.observers = []

    def add_observer(self, observer):
        self.observers.append(observer)

    def remove_observer(self, observer):
        self.observers.remove(observer)

    def get_latest_observer(self):
        return self.observers[-1]

    def notify_observers(self, progress_val, message):
        for observer in self.observers:
            observer.update(progress_val, message)

class ObserverContext:
    def __init__(self, event_handler, observer):
        self.event_handler = event_handler
        self.observer = observer

    def __enter__(self):
        self.event_handler.add_observer(self.observer)
        return self.observer

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.event_handler.remove_observer(self.observer)

@dataclass
class PromptPrefix:
    positive: str
    negative: str

_default_prefixes = PromptPrefix(
    positive = '(masterpiece:1.2), best quality, highres, extremely detailed wallpaper, perfect lighting, (extremely detailed CG:1.2), (8k:1.1), (ultra-detailed), (best illustration), (oil painting), absurdres',
    negative = 'nsfw, easynegative, (bad_prompt_version2:0.8), ng_deepnegative_v1_75t, badhandsv5-neg, censored, (low quality:1.3), (worst quality:1.3), (monochrome:0.8), (deformed:1.3), (malformed hands:1.4), (poorly drawn hands:1.4), (mutated fingers:1.4), (bad anatomy:1.3), (extra limbs:1.35), (poorly drawn face:1.4), (signature:1.2), (artist name:1.2), (watermark:1.2)'
)

def _create_option_payload(sd_model_checkpoint: str, **kwargs) -> Dict[str, Any]:
    defaults = {
        "CLIP_stop_at_last_layers": 2,
        "sd_vae": "orangemix.vae.pt"
    }
    return {**defaults, "sd_model_checkpoint": sd_model_checkpoint, **kwargs}

def _create_payload(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    default_values = {
        "steps": 30,
        "sampler_index": 'DPM++ 2M Karras',
        "cfg_scale": 7,
        "width": 512,
        "height": 512,
        "denoising_strength": 0.45,
        "hr_scale": 2,
        "hr_upscaler": '4x_fatal_Anime_500000_G',
        "hr_second_pass_steps": 15
    }

    if overrides:
        default_values.update(overrides)

    return default_values

@dataclass
class SDModelType:
    option_payload: Dict[str, str]
    payload: Optional[Dict[str, str]] = field(default_factory=_create_payload)
    prompt_prefix: Optional[PromptPrefix] = field(default_factory=_default_prefixes)

    @staticmethod
    def create(
            sd_model_checkpoint: str, 
            option_payload_overrides: Optional[Dict[str, Any]] = None, 
            payload_overrides: Optional[Dict[str, Any]] = None,
            custom_prompt_prefix: Optional[PromptPrefix] = None
    ) -> 'SDModelType':
        option_payload = _create_option_payload(sd_model_checkpoint, **(option_payload_overrides or {}))
        payload = _create_payload(payload_overrides or {})
        prompt_prefix = custom_prompt_prefix if custom_prompt_prefix else _default_prefixes
        return SDModelType(option_payload=option_payload, payload=payload, prompt_prefix=prompt_prefix)

class CheckpointNamesGUI:
    SARDONYX = 'Sardonyx Redux v20 - Stylized/Cartoon'
    REALISTICVISION = 'Realistic Vision v4 - Photorealistic'
    GHOSTMIX = 'Ghost Mix v20 - Anime/Photorealistic'
    AOM3A1B = 'AbyssOrangeMix3 A1B - Anime/Landscapes'
    ANYTHINGV3 = 'Anything v3 - Cartoon/Anime/Drawings'
    KANPIRO = 'KanpiroMix v20 - Photorealistic'
    SCHAUXIER = 'Sch Auxier v10 - Cartoon/Anime'
    DREAMSHAPER = 'Dream Shaper - Photorealistic/Paintings'
    REVANIMATED = "Rev Animated - Anime/Cartoon/Illustration"
    MISTOON = "Mistoon v20 - Cartoon/Anime/Drawings"

class SDModels:
    sardonyxREDUX_v20 = SDModelType.create('sardonyxREDUX_v20.safetensors [40d4f9d626]')
    abyssorangemix3A0M3_aom3alb = SDModelType.create('abyssorangemix3AOM3_aom3a1b.safetensors [5493a0ec49]')
    anything_v3_full = SDModelType.create('anything-v3-full.safetensors [abcaf14e5a]')
    mistoonAnime_v20 = SDModelType.create('mistoonAnime_v20.safetensors [c35e1054c0]')
    kanpiromix_v20 = SDModelType.create('kanpiromix_v20.safetensors [b8cf1eaa89]')
    schAuxier_v10 = SDModelType.create('schAuxier_v10.safetensors [0265299a9d]')
    ghostmix_v20Bakedvae = SDModelType.create('ghostmix_v20Bakedvae.safetensors [e3edb8a26f]',
        option_payload_overrides={
            "sd_vae": "Automatic"
        },)
    realistic_vision_v40_vae = SDModelType.create('realisticVisionV40_v40VAE.safetensors [e9d3cedc4b]',
        option_payload_overrides={
            "sd_vae": "Automatic"
        },
        payload_overrides={
            "sampler_index": 'DPM++ SDE Karras',
            "cfg_scale": 5,
            "denoising_strength": 0.3,
            "hr_upscaler": "R-ESRGAN 4x+",
        },
        custom_prompt_prefix=PromptPrefix(
            positive='RAW photo, subject, (high detailed skin:1.2), 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3', 
            negative='(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime), text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, BadDream'
        )
    )
    dreamshaper_v8_vae = SDModelType.create('dreamshaper_8.safetensors [879db523c3]',
        option_payload_overrides={
            "sd_vae": "Automatic"
        })
    rev_animated_v122 = SDModelType.create('revAnimated_v122.safetensors [4199bcdd14]')

    @staticmethod
    def get_gui_model_names_list():
        return [getattr(CheckpointNamesGUI, attr) for attr in dir(CheckpointNamesGUI) if not attr.startswith("__")]

    @staticmethod
    def get_sd_model(gui_model_name) -> SDModelType:
        # Define a mapping between checkpoint names and their corresponding SDModelType objects
        mapping = {
            CheckpointNamesGUI.SARDONYX: SDModels.sardonyxREDUX_v20,
            CheckpointNamesGUI.AOM3A1B: SDModels.abyssorangemix3A0M3_aom3alb,
            CheckpointNamesGUI.ANYTHINGV3: SDModels.anything_v3_full,
            CheckpointNamesGUI.KANPIRO: SDModels.kanpiromix_v20,
            CheckpointNamesGUI.SCHAUXIER: SDModels.schAuxier_v10,
            CheckpointNamesGUI.GHOSTMIX: SDModels.ghostmix_v20Bakedvae,
            CheckpointNamesGUI.REALISTICVISION: SDModels.realistic_vision_v40_vae,
            CheckpointNamesGUI.DREAMSHAPER: SDModels.dreamshaper_v8_vae,
            CheckpointNamesGUI.REVANIMATED: SDModels.rev_animated_v122,
            CheckpointNamesGUI.MISTOON: SDModels.mistoonAnime_v20
        }
        # Retrieve and return the SDModelType object for the given checkpoint name
        return mapping.get(gui_model_name)

    @staticmethod
    def get_sd_model_name(sd_model: SDModelType):
        checkpoint_title = sd_model.option_payload['sd_model_checkpoint']
        match = re.search(r"^(.*?)(\.safetensors|\.ckpt)", checkpoint_title)
        if match:
            return match.group(1)
        else:
            raise ValueError(f'Could not extract title of model {sd_model}')
