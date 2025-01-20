## Steps to Run SAM with TensorRT

### Step 0: Install Requirements  
Run the following command to install dependencies:  
```bash
pip install -r requirements.txt
```
### Step 1: Download SAM ViT-H Checkpoint
Download the checkpoint from the official repository:
[Segment Anything Model - Checkpoints](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints)

### Step 2: Place the Checkpoint File
Move the downloaded checkpoint file to the following directory:
```bash
models/pth
```

### Step 3: Generate Encoder ONNX File
Run the script to export the encoder model:

```bash
python export_encoder_model.py --quantize=False --device="cpu"
```
### Step 4: Configure TensorRT Paths
Modify your CMakeLists.txt file to replace the paths linking to the bin and lib directories of TensorRT.

### Step 5: Build and Run the SAM Target
Compile and run the SAM target from the build/Debug directory, or modify the path in the main file accordingly.
