import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.append('../lib/sentence-transformers-222/sentence-transformers')
from sentence_transformers import SentenceTransformer, models


if __name__ == "__main__":
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    st_model = SentenceTransformer("../lib/sentence-transformers-222/all-MiniLM-L6-v2")

    df_submission = pd.read_csv( "../lib/stable-diffusion-image-to-prompts/prompts.csv")
    dist = {}
    for num in range(len(df_submission['prompt'])):
        dist[df_submission['imgId'][num]] = df_submission['prompt'][num]
    print(dist)