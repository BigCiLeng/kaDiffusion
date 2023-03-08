import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.append('lib/sentence-transformers-222/sentence-transformers')
from sentence_transformers import SentenceTransformer, models


if __name__ == "__main__":
    st_model = SentenceTransformer("lib/sentence-transformers-222/all-MiniLM-L6-v2")

    df_submission = pd.read_csv( "lib/stable-diffusion-image-to-prompts/prompts.csv")
    dist = {}
    for num in range(len(df_submission['prompt'])):
        dist[df_submission['imgId'][num]] = df_submission['prompt'][num]
    
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    score = 0.0
    # clip prompt
    clip_interrogator_file = open('data_output/clip_interrogator_outputs.txt', 'r')
    while True:
        line = clip_interrogator_file.readline()
        if line:
            image_name, prompts = line.split(':')[0], line.split(':')[-1]
            
            clip_interrogator_temp_embeddings = torch.Tensor(st_model.encode(prompts).flatten()).unsqueeze(0)
            right_embeddings = torch.Tensor(st_model.encode(dist[image_name]).flatten()).unsqueeze(0)

            score += cos(clip_interrogator_temp_embeddings, right_embeddings)

        else:
            break
    # ofa embeddings
    
    
    print(score / len(df_submission['prompt']))