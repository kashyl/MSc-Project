import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Any

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(ROOT_DIR, 'img')
SD_URL = "http://127.0.0.1:7860"

@dataclass
class PromptPrefix:
    positive: str
    negative: str

_default_prefixes = PromptPrefix(
    positive = '(masterpiece:1.2), best quality, highres, extremely detailed wallpaper, perfect lighting, (extremely detailed CG:1.2), (8k:1.1), (ultra-detailed), (best illustration), (oil painting), absurdres',
    negative = 'easynegative, (bad_prompt_version2:0.8), ng_deepnegative_v1_75t, badhandsv5-neg, censored, (low quality:1.3), (worst quality:1.3), (monochrome:0.8), (deformed:1.3), (malformed hands:1.4), (poorly drawn hands:1.4), (mutated fingers:1.4), (bad anatomy:1.3), (extra limbs:1.35), (poorly drawn face:1.4), (signature:1.2), (artist name:1.2), (watermark:1.2)'
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

class SDModels:
    sardonyxREDUX_v20 = SDModelType.create('sardonyxREDUX_v20.safetensors [40d4f9d626]')
    abyssorangemix3A0M3_aom3 = SDModelType.create('abyssorangemix3AOM3_aom3.safetensors [d124fcl8f0]')
    abyssorangemix3A0M3_aom3alb = SDModelType.create('abyssorangemix3AOM3_aom3a1b.safetensors [5493a0ec49]')
    anyloraCheckpoint_novaeFpl6 = SDModelType.create('anyloraCheckpoint_novaeFp16.safetensors [ad1150a839]')
    anything_v3_full = SDModelType.create('anything-v3-full.safetensors [abcafl4e5a]')
    breakdro_11464 = SDModelType.create('breakdro_I1464.safetensors [683fb86a54]')
    mistoonAnime_v20 = SDModelType.create('mistoonAnime_v20.safetensors [c35e1054c0]')
    kanpiromix_v20 = SDModelType.create('kanpiromix_v20.safetensors [b8cfleaa89]')
    walnutcreamBlend_herbmixV1 = SDModelType.create('walnutcreamBlend_herbmixV1.safetensors [09524d894d]')
    schAuxier_v10 = SDModelType.create('schAuxier_v10.safetensors [0265299a9d]')
    ghostmix_v20Bakedvae = SDModelType.create('ghostmix_v20Bakedvae.safetensors [e3edb8a26f]',
        option_payload_overrides={
            "sd_vae": "Automatic"
        },)
    perfectWorld_v4Baked = SDModelType.create('perfectWorld_v4Baked.safetensors [24a393500f]',
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


# prompt reference:
# ------------------------
# "prompt": prompt_full,
# "negative_prompt": prompt_negative,
# "steps": 5,
# "enable_hr": False,
# "denoising_strength": 0,
# "firstphase_width": 0,
# "firstphase_height": 0,
# "hr_scale": 2,
# "hr_upscaler": "string",
# "hr_second_pass_steps": 0,
# "hr_resize_x": 0,
# "hr_resize_y": 0,
# "styles": [
#     "string"
# ],
# "seed": -1,
# "subseed": -1,
# "subseed_strength": 0,
# "seed_resize_from_h": -1,
# "seed_resize_from_w": -1,
# "sampler_name": "string",
# "batch_size": 1,
# "n_iter": 1,
# "cfg_scale": 7,
# "width": 512,
# "height": 512,
# "restore_faces": False,
# "tiling": False,
# "do_not_save_samples": False,
# "do_not_save_grid": False,
# "eta": 0,
# "s_min_uncond": 0,
# "s_churn": 0,
# "s_tmax": 0,
# "s_tmin": 0,
# "s_noise": 1,
# "override_settings": {},
# "override_settings_restore_afterwards": True,
# "script_args": [],
# "sampler_index": "Euler",
# "script_name": "string",
# "send_images": True,
# "save_images": False,
# "alwayson_scripts": {}


"""
Options:

{
  "samples_save": true,
  "samples_format": "png",
  "samples_filename_pattern": "",
  "save_images_add_number": true,
  "grid_save": true,
  "grid_format": "png",
  "grid_extended_filename": false,
  "grid_only_if_multiple": true,
  "grid_prevent_empty_spots": false,
  "n_rows": -1,
  "enable_pnginfo": true,
  "save_txt": false,
  "save_images_before_face_restoration": false,
  "save_images_before_highres_fix": false,
  "save_images_before_color_correction": false,
  "save_mask": false,
  "save_mask_composite": false,
  "jpeg_quality": 80,
  "webp_lossless": false,
  "export_for_4chan": true,
  "img_downscale_threshold": 4,
  "target_side_length": 4000,
  "img_max_size_mp": 200,
  "use_original_name_batch": true,
  "use_upscaler_name_as_suffix": false,
  "save_selected_only": true,
  "save_init_img": false,
  "temp_dir": "",
  "clean_temp_dir_at_start": false,
  "outdir_samples": "",
  "outdir_txt2img_samples": "outputs/txt2img-images",
  "outdir_img2img_samples": "outputs/img2img-images",
  "outdir_extras_samples": "outputs/extras-images",
  "outdir_grids": "",
  "outdir_txt2img_grids": "outputs/txt2img-grids",
  "outdir_img2img_grids": "outputs/img2img-grids",
  "outdir_save": "log/images",
  "outdir_init_images": "outputs/init-images",
  "save_to_dirs": true,
  "grid_save_to_dirs": true,
  "use_save_to_dirs_for_ui": false,
  "directories_filename_pattern": "[date]",
  "directories_max_prompt_words": 8,
  "ESRGAN_tile": 192,
  "ESRGAN_tile_overlap": 8,
  "realesrgan_enabled_models": [
    "R-ESRGAN 4x+",
    "R-ESRGAN 4x+ Anime6B"
  ],
  "upscaler_for_img2img": "Unknown Type: null",
  "SCUNET_tile": 256,
  "SCUNET_tile_overlap": 8,
  "face_restoration_model": "CodeFormer",
  "code_former_weight": 0.5,
  "face_restoration_unload": false,
  "show_warnings": false,
  "memmon_poll_rate": 8,
  "samples_log_stdout": false,
  "multiple_tqdm": true,
  "print_hypernet_extra": false,
  "unload_models_when_training": false,
  "pin_memory": false,
  "save_optimizer_state": false,
  "save_training_settings_to_txt": true,
  "dataset_filename_word_regex": "",
  "dataset_filename_join_string": " ",
  "training_image_repeats_per_epoch": 1,
  "training_write_csv_every": 500,
  "training_xattention_optimizations": false,
  "training_enable_tensorboard": false,
  "training_tensorboard_save_images": false,
  "training_tensorboard_flush_every": 120,
  "sd_model_checkpoint": "string",
  "sd_checkpoint_cache": 0,
  "sd_vae_checkpoint_cache": 0,
  "sd_vae": "Automatic",
  "sd_vae_as_default": true,
  "inpainting_mask_weight": 1,
  "initial_noise_multiplier": 1,
  "img2img_color_correction": false,
  "img2img_fix_steps": false,
  "img2img_background_color": "#ffffff",
  "enable_quantization": false,
  "enable_emphasis": true,
  "enable_batch_seeds": true,
  "comma_padding_backtrack": 20,
  "CLIP_stop_at_last_layers": 1,
  "upcast_attn": false,
  "randn_source": "GPU",
  "use_old_emphasis_implementation": false,
  "use_old_karras_scheduler_sigmas": false,
  "no_dpmpp_sde_batch_determinism": false,
  "use_old_hires_fix_width_height": false,
  "dont_fix_second_order_samplers_schedule": false,
  "interrogate_keep_models_in_memory": false,
  "interrogate_return_ranks": false,
  "interrogate_clip_num_beams": 1,
  "interrogate_clip_min_length": 24,
  "interrogate_clip_max_length": 48,
  "interrogate_clip_dict_limit": 1500,
  "interrogate_clip_skip_categories": [],
  "interrogate_deepbooru_score_threshold": 0.5,
  "deepbooru_sort_alpha": true,
  "deepbooru_use_spaces": false,
  "deepbooru_escape": true,
  "deepbooru_filter_tags": "",
  "extra_networks_default_view": "cards",
  "extra_networks_default_multiplier": 1,
  "extra_networks_card_width": 0,
  "extra_networks_card_height": 0,
  "extra_networks_add_text_separator": " ",
  "sd_hypernetwork": "None",
  "return_grid": true,
  "return_mask": false,
  "return_mask_composite": false,
  "do_not_show_images": false,
  "send_seed": true,
  "send_size": true,
  "font": "",
  "js_modal_lightbox": true,
  "js_modal_lightbox_initially_zoomed": true,
  "js_modal_lightbox_gamepad": true,
  "js_modal_lightbox_gamepad_repeat": 250,
  "show_progress_in_title": true,
  "samplers_in_dropdown": true,
  "dimensions_and_batch_together": true,
  "keyedit_precision_attention": 0.1,
  "keyedit_precision_extra": 0.05,
  "keyedit_delimiters": ".,\\/!?%^*;:{}=`~()",
  "quicksettings_list": [
    "sd_model_checkpoint"
  ],
  "hidden_tabs": [],
  "ui_reorder": "inpaint, sampler, checkboxes, hires_fix, dimensions, cfg, seed, batch, override_settings, scripts",
  "ui_extra_networks_tab_reorder": "",
  "localization": "None",
  "gradio_theme": "Default",
  "add_model_hash_to_info": true,
  "add_model_name_to_info": true,
  "add_version_to_infotext": true,
  "disable_weights_auto_swap": true,
  "show_progressbar": true,
  "live_previews_enable": true,
  "show_progress_grid": true,
  "show_progress_every_n_steps": 10,
  "show_progress_type": "Approx NN",
  "live_preview_content": "Prompt",
  "live_preview_refresh_period": 1000,
  "hide_samplers": [],
  "eta_ddim": 0,
  "eta_ancestral": 1,
  "ddim_discretize": "uniform",
  "s_churn": 0,
  "s_min_uncond": 0,
  "s_tmin": 0,
  "s_noise": 1,
  "eta_noise_seed_delta": 0,
  "always_discard_next_to_last_sigma": false,
  "uni_pc_variant": "bh1",
  "uni_pc_skip_type": "time_uniform",
  "uni_pc_order": 3,
  "uni_pc_lower_order_final": true,
  "postprocessing_enable_in_main_ui": [],
  "postprocessing_operation_order": [],
  "upscaling_max_images_in_cache": 5,
  "disabled_extensions": [],
  "disable_all_extensions": "none",
  "restore_config_state_file": "",
  "sd_checkpoint_hash": ""
}
"""
