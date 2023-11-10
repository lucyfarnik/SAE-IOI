#%%
# The models from Hoagy et al contain a list of models trained with different
# hyperparams, but that makes the files too big to upload conveniently to GitHub.
# Here I'm extracting only the one model that I'm actually using right now.
import torch as T

sae = T.load("models/sae_l3_r4.pt")[5][0]

T.save(sae, "models/sae_l3_r4_single.pt")

#%%