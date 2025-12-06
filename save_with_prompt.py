
import os
import json
import numpy as np
from PIL import Image
from datetime import datetime
import folder_paths

class SaveImageWithPrompt:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "filename_prefix": ("STRING", {"default": "ComfyUI_Custom"}),
                "save_prompt": ("BOOLEAN", {"default": True, "label_on": "enable", "label_off": "disable"}),
                "save_extra_pnginfo": ("BOOLEAN", {"default": True, "label_on": "enable", "label_off": "disable"}),
            },
            "hidden": {
                "prompt": "PROMPT", 
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "Custom/Save"

    def save_images(self, images, filename_prefix="ComfyUI_Custom", save_prompt=True, save_extra_pnginfo=True, prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        
        # Prepare metadata to save in JSON
        metadata = {}
        if save_prompt and prompt is not None:
            metadata["prompt"] = prompt
        if save_extra_pnginfo and extra_pnginfo is not None:
            metadata["extra_pnginfo"] = extra_pnginfo

        # Save metadata to a JSON file if there is any (and if requested)
        if metadata:
            metadata_filename = f"{filename}_{counter:05}_metadata.json"
            metadata_path = os.path.join(full_output_folder, metadata_filename)
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)

        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
            file = f"{filename}_{counter:05}_.png"
            
            img.save(os.path.join(full_output_folder, file), optimize=True)
            
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return { "ui": { "images": results } }
