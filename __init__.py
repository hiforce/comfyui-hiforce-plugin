import importlib
import os
import sys
import traceback
from pathlib import Path
import folder_paths

here = Path(__file__).parent.resolve()
comfy_path = folder_paths.base_path

sys.path.insert(0, str(Path(here, "").resolve()))
sys.path.insert(0, str(Path(here, "src").resolve()))
for pkg_name in os.listdir(str(Path(here, "src"))):
    sys.path.insert(0, str(Path(here, "src", pkg_name).resolve()))

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = '1'


def load_nodes():
    shorted_errors = []
    full_error_messages = []
    node_class_mappings = {}
    node_display_name_mappings = {}

    for filename in (here / "node_wrappers").iterdir():

        module_name = filename.stem
        try:
            module = importlib.import_module(
                f".node_wrappers.{module_name}", package=__package__
            )
            node_class_mappings.update(getattr(module, "NODE_CLASS_MAPPINGS"))
            if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
                node_display_name_mappings.update(getattr(module, "NODE_DISPLAY_NAME_MAPPINGS"))
        except AttributeError:
            pass  # wip nodes
        except Exception:
            error_message = traceback.format_exc()
            full_error_messages.append(error_message)
            error_message = error_message.splitlines()[-1]
            shorted_errors.append(
                f"Failed to import module {module_name} because {error_message}"
            )

    if len(shorted_errors) > 0:
        full_err_log = '\n\n'.join(full_error_messages)
        print(f"\n\nFull error log from comfyui-hiforce-plugin: \n{full_err_log}\n\n")
    return node_class_mappings, node_display_name_mappings


HF_NODE_MAPPINGS, HF_DISPLAY_NAME_MAPPINGS = load_nodes()

NODE_CLASS_MAPPINGS = {
    **HF_NODE_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **HF_DISPLAY_NAME_MAPPINGS,
}
