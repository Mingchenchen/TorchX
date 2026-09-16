import sys
import types
import torchfold


def patch_pkl_remapping(pkl_name):
    if not pkl_name:
        print(">>> [Patch] pkl_name is empty, skipping remapping.")
        return
    """Completely remap module path to torchfold to resolve old cache loading issues."""

    # 1. Map top-level package
    sys.modules[pkl_name] = torchfold

    # 2. Map key submodules (manually specify the paths most prone to errors as needed)
    from torchfold.structure import tables
    from torchfold import processing

    # Create necessary intermediate layers
    struct_mod = types.ModuleType('structure')
    sys.modules[f'{pkl_name}.structure'] = struct_mod
    sys.modules[f'{pkl_name}.structure.structure_tables'] = tables

    # Map model/processing layer
    sys.modules[f'{pkl_name}.model'] = processing

    # 3. Dynamically mirror all submodules
    # Iterate over all currently loaded torchfold submodules and create corresponding mirrors
    for mod_name, mod_obj in list(sys.modules.items()):
        if mod_name.startswith('torchfold.'):
            fake_name = mod_name.replace('torchfold.', f'{pkl_name}.')
            if fake_name not in sys.modules:
                sys.modules[fake_name] = mod_obj

    print(f">>> [Patch] {pkl_name} compatibility layer initialized.")
