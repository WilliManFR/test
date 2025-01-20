Step 0: install requirements.txt
Step 1: Donwload sam vit-h checkpoint: https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints
Step 2: Place the file in `models\pth`
Step 3: Generate encoder onnx file with `export_encoder_model` script: `python export_encoder_model.py --quantize=False --device="cpu"`
Step 4: replace your own path to link bin and lib directory of TensorRT in CMakelists.txt
Step 5: Build and run sam target from build/Debug orelse change path in the main