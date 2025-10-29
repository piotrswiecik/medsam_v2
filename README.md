# MedSAM pipeline v.1.0

## Usage

```shell
# training
uv run main.py --config=config.file.name.json

# inference
uv run main.py --config=config.dev.json --mode=predict --image_path="./data/arcade/syntax/test/images/1.png" --checkpoint_path="/Users/piotrswiecik/dev/ives/coronary/medsam_pipeline/data/model/best_model_multihead_200ep.pth
```

## clahe testing

```
uv run preprocessing.py /Users/piotrswiecik/dev/ives/coronary/datasets/wum_v2/43_I0508338.VIM.DCM.18.png
```