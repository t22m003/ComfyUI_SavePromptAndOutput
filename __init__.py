
from .save_with_prompt import SaveImageWithPrompt
from .hooks import apply_hook

# Apply the execution hook
apply_hook()

NODE_CLASS_MAPPINGS = {
    "SaveImageWithPrompt": SaveImageWithPrompt
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveImageWithPrompt": "Save Image With Prompt"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
