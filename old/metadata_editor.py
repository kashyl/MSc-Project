import os, json, re, copy
from PIL import Image
from PIL.PngImagePlugin import PngInfo


DIR_PATH = os.path.join('')
BASE_OUTPUT_DIR = os.path.join(DIR_PATH, 'output')

CIVITAI_LORA_HASHES = {}

CIVITAI_MODEL_HASHES = {}

CIVITAI_META_TEMPLATE = {}


def getImagePaths():
    paths = []
    for file in os.listdir(DIR_PATH):
        if file.endswith(".png"):
            paths.append(os.path.join(DIR_PATH, file))
    return paths

def makeDir():
    base_directory = BASE_OUTPUT_DIR

    # Check if the base directory exists, create it if not
    if not os.path.exists(base_directory):
        os.makedirs(base_directory)

    # Find the highest numbered subdirectory
    subdirectories = [name for name in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, name))]
    highest_number = max([int(subdir) for subdir in subdirectories]) if subdirectories else -1

    # Increment the highest number
    new_directory = os.path.join(base_directory, str(highest_number + 1))

    # Create the new directory
    os.makedirs(new_directory)

    # Return the path of the new directory
    return(new_directory)

def saveImage(image_path, target_image, metadata, output_dir):
    new_image_name = os.path.basename(image_path)
    new_image_path = os.path.join(output_dir, new_image_name)
    target_image.save(new_image_path, pnginfo=metadata)

def getResourceHash(resource_type_and_name:str):
    try:
        res_hash = CIVITAI_LORA_HASHES[resource_type_and_name]
        return res_hash
    except KeyError:
        raise KeyError(f'WARNING: Could not find hash for resource "{resource_type_and_name}"')

def getPromptResources(params:str):
    res_metadata = {}
    resources_types_and_names = re.findall(fr"<(\w+?:.*?)\:", params)
    for resource in resources_types_and_names:
        try:
            res_metadata[resource] = getResourceHash(resource)
        except KeyError as e:
            print(e)
            warnings.append(e)
            continue

    return res_metadata

def generateResourcesMetadata(prompt_resources:dict, meta_template:dict):
    resources_metadata = copy.deepcopy(meta_template)

    for r_type_and_name, r_hash in prompt_resources.items():
        resources_metadata[f'{r_type_and_name}'] = r_hash
    return resources_metadata

def removeExistingHashes(params):
    pattern = r', Hashes: {.*?}'
    result = re.sub(pattern, '', params)
    return result

def addCivitaiMetadataToParams(params, civitai_resources_meta):
    res_meta_str = json.dumps(civitai_resources_meta)
    return params + f', Hashes: {res_meta_str}'

def generateNewMetadata(params:str, res_meta:str):
    params = removeExistingHashes(params)
    metadata_with_resources = addCivitaiMetadataToParams(params=params, civitai_resources_meta=res_meta)    # Append civitai resources metadata to new image metadata
    pnginfo= PngInfo()
    pnginfo.add_text("parameters", metadata_with_resources) # Create new metadata to be used for the image
    return pnginfo

def removeGenerationInfo(metadata:str):
    # Find the index of "Steps: 30"
    steps_index = metadata.find("\nSteps:")

    # Find the index of "Seed: "
    seed_index = metadata.find("Seed: ")

    # Remove the positive and negative prompt parameters
    metadata = metadata[steps_index:]

    # Remove the "Seed" if it exists
    if seed_index != -1:
        metadata = re.sub(r" Seed:\s\d+,", '', metadata)

    return metadata

def get_model_hash(text):
    pattern = r"Model hash: (\w+)"
    match = re.search(pattern, text)
    return match.group(1)

image_paths = getImagePaths()
print(f'Adding resource metadata to {len(image_paths)} images.')

output_dir = makeDir()
warnings = []

for i, path in enumerate(image_paths):
    
    target_image = Image.open(path)
    metadata = target_image.info
    parameters = metadata['parameters']

    prompt_res = getPromptResources(parameters)                 # Get resources types and names from prompt
    meta_template = CIVITAI_META_TEMPLATE
    meta_template["model"] = get_model_hash(parameters)
    res_meta = generateResourcesMetadata(prompt_res, meta_template)          # Generate metadata dict with resource type:name key and hash value

    parameters = removeGenerationInfo(parameters)

    new_metadata = generateNewMetadata(parameters, res_meta)

    saveImage(image_path=path, target_image=target_image, metadata=new_metadata, output_dir=output_dir)

    print(f'Added resources metadata "{prompt_res}" to image "{os.path.basename(path)}" ({i + 1} out of {len(image_paths)}).')


print(f'Finished adding Civitai resource metadata to {len(image_paths)} images.')

if warnings:
    print(f'{len(warnings)} occurred during resource metadata addition:')
    for warning in warnings:
        print(warning)
