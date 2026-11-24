import re
import torch
import timm
import numpy as np
from pathlib import Path
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import login
import cv2
import openslide
import copy
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
import torch.nn as nn
import torch.nn.functional as F
import torch as t
import pandas as pd
DEVICE = "cuda"
login("")
size = 256
img = openslide.OpenSlide("polymer/test_image/testing.svs")
pathh = Path("/polymer/infer/registered")
pathh.mkdir(parents=True, exist_ok=True)
width, height = img.dimensions

for y in range(0, height - size + 1, size):
    for x in range(0, width - size + 1, size):
        rgb = np.array(
            img.read_region((x, y), 0, (size, size)).convert("RGB"),
            copy=True,
        )
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) 
        _, otsu = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        rgb[otsu == 0] = 255
        tissue_fraction = np.mean(otsu > 0)
        if tissue_fraction < 0.20:
            continue
        cv2.imwrite(str(pathh / f"HE_CROP_x{x}_y{y}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 9])
    

timm_kwargs = {
    "img_size": 224,
    "patch_size": 14,
    "depth": 24,
    "num_heads": 24,
    "init_values": 1e-5,
    "embed_dim": 1536,
    "mlp_ratio": 2.66667 * 2,
    "num_classes": 0,
    "no_embed_class": True,
    "mlp_layer": timm.layers.SwiGLUPacked,
    "act_layer": torch.nn.SiLU,
    "reg_tokens": 8,
    "dynamic_img_size": True,
}

device = torch.device("cuda")

model = timm.create_model(
    "hf-hub:MahmoodLab/UNI2-h",
    pretrained=True,
    **timm_kwargs
).to(device)

transform = create_transform(
    **resolve_data_config(model.pretrained_cfg, model=model)
)

model.eval()

root = Path("/fs/scratch/PAS2942/Alejandro/polymer/infer/registered")
outdir = root / "embeddings"
outdir.mkdir(parents=True, exist_ok=True)

coord_pattern = re.compile(r"_x(\d+)_y(\d+)")

for folder in [root]:

    if not folder.is_dir():
        continue

    pathos = folder

    if not pathos.exists():
        continue

    embeddings = []
    coords = []

    for file in pathos.iterdir():

        if file.suffix.lower() not in {".png", ".jpg", ".jpeg", ".svs"}:
            continue

        match = coord_pattern.search(file.stem)

        if match is None:
            print(f"Could not parse coordinates: {file.name}")
            continue

        x = int(match.group(1))
        y = int(match.group(2))

        with Image.open(file) as image:
            image = transform(image.convert("RGB")).unsqueeze(0).to(device)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            feature_emb = model(image)

        embeddings.append(feature_emb.squeeze(0).float().cpu().numpy())
        coords.append([x, y])
        
        del image
        del feature_emb

    slide_data = {}
    for coord, embedding in zip(coords, embeddings):
        coord = tuple(coord)
        if coord in slide_data:
            raise ValueError(f"duplicate coordinates")
        slide_data[coord] = {"patch_embeddings": embedding}

    output_path = outdir / f"{folder.name}.npy"
    np.save(output_path, slide_data)

    del embeddings
    del coords
    del slide_data

def edge_index_cosine_similarity(x, threshold = 0.5):
    x_normalized = F.normalize(x, p=2, dim=1, eps=1e-8)
    similarity = x_normalized @ x_normalized.t()
    edge_index = t.nonzero(similarity > threshold).t().contiguous()
    return edge_index

class Model(nn.Module):
    def __init__(self, input_dim = 1536, hidden_dim = 256, output_dim = 9):
        super(Model, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads = 4)
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.5), nn.Linear(hidden_dim, output_dim))
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = x.unsqueeze(1)
        x, _ = self.attention(x,x,x)
        x = x.squeeze(1)
        x= x.mean(dim=0)
        x = self.decoder(x)
        return x 

model = Model(input_dim=1536, hidden_dim=256, output_dim=9).to(DEVICE)

model.load_state_dict(
    t.load("results/fold_1_best_model.pt", map_location=DEVICE, weights_only=True)
)

model.eval()
data = np.load(
    "/registered/embeddings/registered.npy",
    allow_pickle=True,
).item()
patches = []
for coord in list(data.keys())[:2048]:
    patches.append(data[coord]["patch_embeddings"])

patches = np.asarray(patches, dtype=np.float32)
patches = t.from_numpy(patches).to(DEVICE)

with t.inference_mode():
    edge_index = edge_index_cosine_similarity(patches)
    results = model(patches, edge_index).cpu().numpy()

print(f"Collagen IV:        {results[0]:.4f}")
print(f"Collagen VI:        {results[1]:.4f}")
print(f"Fibrillar collagen: {results[2]:.4f}")
print(f"Fibronectin:        {results[3]:.4f}")
print(f"Laminin:            {results[4]:.4f}")
print(f"Osteopontin:        {results[5]:.4f}")
print(f"Tenascin C:         {results[6]:.4f}")
print(f"Thrombospondin:     {results[7]:.4f}")
print(f"Vitronectin:        {results[8]:.4f}")
