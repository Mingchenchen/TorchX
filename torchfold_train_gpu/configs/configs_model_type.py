# torchfold model-type registry

model_configs = {
    "torchfold_af3_default": {
        "model_class": "torchfold.model.alphafold3:AlphaFold3",
        "param_format": "af3_bin_zst",
    },
    # ---- template for an ENTIRELY DIFFERENT model (inactive example) ----
    # Register the new model class + its config overrides (deep-merged onto
    # configs_base), then launch with `--model_name my_model_v1`. build_model
    # imports model_class and builds it from the merged config; provide a matching
    # param loader keyed by `param_format`.
    #
    # "my_model_v1": {
    #     "model_class": "mypkg.models:MyModel",
    #     "param_format": "torch_state_dict",
    #     "c_z": 256,
    #     "model": {
    #         "pairformer": {"c_z": 256},
    #         "diffusion_module": {"c_z": 256},
    #     },
    # },
}
