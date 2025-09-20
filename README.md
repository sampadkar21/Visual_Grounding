# Visual Grounding with DINOv3 and BGE

This repository contains the implementation of a Visual Grounding model that can localize objects in an image based on a natural language text prompt. The model identifies the region of interest described by the text and predicts a bounding box around it. The architecture leverages powerful pre-trained models: DINOv3 as the vision backbone and BGE-Small-EN-v1.5 as the text encoder. These two modalities are fused using a Transformer Decoder, which then predicts the bounding box coordinates.

## Key Features
1. **Text-to-Object Localization:** Takes an image and a text prompt to find a specific object.
2. **Powerful Encoders:** Utilizes frozen, state-of-the-art vision (DINOv3) and text (BGE) encoders for robust feature extraction.
3. **Transformer-based Fusion:** Employs a Transformer Decoder to effectively combine visual and textual information.
4. **End-to-End Training:** Trained on the RefCOCO dataset for referential expression grounding.

## Model Architecture
The model is composed of three main parts:

- **Vision Encoder:** A frozen `facebook/dinov3-vitb16-pretrain-lvd1689m` model processes the input image to extract rich visual features.
- **Text Encoder:** A frozen `BAAI/bge-small-en-v1.5` model processes the text prompt to generate contextual text embeddings.
- **Fusion and Prediction:**
  - Linear projection layers align the dimensions of the image and text embeddings.
  - A 6-layer Transformer Decoder takes the text embeddings as the query (`tgt`) and the image embeddings as the memory (`memory`) to fuse the information.
  - The output from the decoder is passed through a final MLP head to predict the four bounding box coordinates (`xmin`, `ymin`, `xmax`, `ymax`).

### Model Summary
- **Total Parameters:** 144,968,836  
- **Trainable Parameters:** 25,948,420

## Dataset
The model was trained on the RefCOCO dataset, a standard benchmark for referential expression grounding. The dataset was loaded using the Hugging Face `datasets` library.

**Dataset Splits:**
```python
DatasetDict({
    train: Dataset({
        features: ['file_name', 'raw_sentences', 'bbox'],
        num_rows: 42404
    }),
    validation: Dataset({
        features: ['file_name', 'raw_sentences', 'bbox'],
        num_rows: 3811
    }),
    test: Dataset({
        features: ['file_name', 'raw_sentences', 'bbox'],
        num_rows: 1975
    }),
    testB: Dataset({
        features: ['file_name', 'raw_sentences', 'bbox'],
        num_rows: 1810
    })
})
```
After processing sentences, the final sample counts were:
- Train Samples: 120,624
- Validation Samples: 10,834
- Test Samples: 10,752

### Training & Evaluation

The model was trained for 10 epochs using the AdamW optimizer and a cosine learning rate scheduler. The loss was calculated using the distance_box_iou_loss.

**Training Results**:
```table
Epoch	Train Loss	Train IoU	Val Loss	Val IoU
1	 0.6769	0.3658	0.6040	            0.4308
2	0.5558	0.4791	0.5361	            0.4991
3	0.5027	0.5302	0.5003	            0.5329
4	0.4678	0.5628	0.4795	            0.5525
5	0.4392	0.5892	0.4518	            0.5778
6	0.4104	0.6155	0.4352	            0.5940
7	0.3856	0.6380	0.4212	            0.6065
8	0.3655	0.6562	0.4093	            0.6174
9	0.3509	0.6696	0.4065	            0.6197
10	0.3440	0.6759	0.4056	      0.6205
```
### Final Test Results:
- Mean IoU (mIoU): 0.6196
- Mean Average Precision (MAP):
  - mAP@50: 0.5746
  - mAP@75: 0.2173

### Inference & Demo
To use the trained model, load the state dictionary and use the predict_and_crop function provided in the notebook.

**Requirements**:
```bash pip install torch torchvision transformers datasets albumentations torchmetrics Pillow requests```

**Inference Code**:
```python
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn

# --- 1. Load Model Components ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 224
DIM = 512

# Load models and tokenizer
img_model = AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
text_model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5")
decoder_layer = nn.TransformerDecoderLayer(d_model=DIM, nhead=8, batch_first=True)
decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

# --- 2. Initialize and Load Trained Model ---
inf_model = GroundingModel(img_model, text_model, decoder, DIM).to(device)
inf_model_path = 'path/to/your/best_model_iou.pt'
inf_model.load_state_dict(torch.load(inf_model_path, map_location=device))
inf_model.eval()

# --- 3. Run Inference ---
image_url = "https://bouldervet.com/wp-content/uploads/2023/09/dog-cat-coexistence-1024x683.jpg"
prompt = "the cat on the sofa"

# The 'transforms' object should also be defined as in the notebook
cropped_object, box, image_with_box = predict_and_crop(
    image_url, prompt, inf_model, tokenizer, transforms, device
)

# Display the results
if cropped_object:
    image_with_box.show()
    cropped_object.show()
```

### Demo Example
