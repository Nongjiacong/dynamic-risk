This folder supports two LightGBM model files.

## 1. Private manuscript model

Place the real manuscript LightGBM model file in this folder with the exact filename:

`lightgbm_model.txt`

The file should be trained on the required landmark-level feature columns documented in the repository README.

If this file exists, the package will prioritize it over the bundled model.

## 2. Bundled public model

The repository may also include:

`bundled_lightgbm_model.txt`

This file is safe for public release and is intended only for open-source use, smoke tests, and walkthroughs. It should not be presented as the manuscript model.

If the manuscript model is missing but the bundled model exists, the package will load it.

If neither model file exists, the package will use the built-in reference scorer instead.

## Optional metadata

You can also add:

`model_metadata.json`

to configure the public model name, version, description, intended use text, and alert threshold. A starter example is provided in `model_metadata.example.json`.
