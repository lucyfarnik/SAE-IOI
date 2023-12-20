#%%
# Proper comments are here: https://docs.google.com/document/d/1fLUjUzk5Bze8ZCqosTb31Inixv8KaV6hiHFscr36nJ0/edit#heading=h.vyhiz35mq1n7
import torch as T
import numpy as np
from transformer_lens import HookedTransformer, ActivationCache, patching, utils
from tqdm import tqdm
from functools import partial
import pandas as pd
import plotly.express as px
from jaxtyping import Float
import einops

device = T.device("mps" if T.backends.mps.is_available() else "cpu")
T.manual_seed(42)
np.random.seed(42)
#%%
gpt = HookedTransformer.from_pretrained("gpt2-small").to(device)
gpt.set_use_attn_result(True)

#%%
# quick sanity check
prompt = "When Mary and John went to the store, John gave a drink to"
logits, activations = gpt.run_with_cache(prompt)

"Most likely next token:" + gpt.to_string(logits[0, -1].argmax())

# %%
sae = T.load("models/sae_l3_r4_single.pt")
sae.to_device(device)
# %%
sea_acts = sae.encode(activations['resid_post', 3].squeeze())
# %%
# find the duplicate tokens indices
prompt_str_tokens = gpt.to_str_tokens(prompt)
already_seen = set()
duplicate_indices = []
nonduplicate_indices = []
for i, token in enumerate(prompt_str_tokens):
    if token in already_seen:
        duplicate_indices.append(i)
    else:
        nonduplicate_indices.append(i)
        already_seen.add(token)
duplicate_indices
# %%
# here's the rough setup I'm thinking:
# 1. get the SAE activations on a ton of tokens, split them based on whether it's a duplicate
# 2. get the mean of each of these two
# 3. find the SAE features where the means differ the most
# KEEP IN MIND right now this is a sample size of 1 prompt
sea_acts_duplicates = sea_acts[duplicate_indices]
sea_acts_duplicates.shape
# %%
sea_acts_nonduplicates = sea_acts[nonduplicate_indices]
sea_acts_nonduplicates.shape
# %%
mean_diff = sea_acts_duplicates.mean(axis=0) - sea_acts_nonduplicates.mean(axis=0)
mean_diff.abs().topk(10)

# %%
# okay, now let's do the same thing but with more prompts
#%%
# generate a bunch of random prompts, have some of the tokens be duplicates
dataset = []
dataset_size = 1000
prompt_length = 100
d_vocab = gpt.cfg.d_vocab
for _ in range(dataset_size):
    prompt = []
    duplicate_indices = []
    nonduplicate_indices = []
    for i in range(prompt_length):
        if i>0 and np.random.random() < 0.5:
            # duplicate
            token = prompt[np.random.randint(i)]
            prompt.append(token)
            duplicate_indices.append(i)
        else:
            # nonduplicate
            token = np.random.randint(d_vocab)
            prompt.append(token)
            nonduplicate_indices.append(i)
    dataset.append((
        prompt,
        {"duplicate_indices": duplicate_indices,
         "nonduplicate_indices": nonduplicate_indices},
    ))
# %%
dataset_prompts = T.tensor([prompt for prompt, _ in dataset]).to(device)
dataset_prompts.shape
# %%
# need to split this up to avoid OOM errors
batch_size = 10
sae_duplicate_acts = []
sae_nonduplicate_acts = []
for subset_i in tqdm(range(dataset_size//batch_size)):
    batch = dataset_prompts[subset_i*10:(subset_i+1)*10]
    # prepend the BOS token
    batch = T.cat([T.ones(batch.shape[0], 1).long().to(device)*50256, batch], dim=1)

    _, activations = gpt.run_with_cache(batch)
    for i in range(batch_size):
        prompt, indices = dataset[subset_i*10+i]
        sea_acts = sae.encode(activations['resid_post', 3][i, 1:])
        sae_duplicate_acts.append(sea_acts[indices['duplicate_indices']])
        sae_nonduplicate_acts.append(sea_acts[indices['nonduplicate_indices']])
sae_duplicate_acts = T.cat(sae_duplicate_acts)
sae_nonduplicate_acts = T.cat(sae_nonduplicate_acts)
sae_all_acts = T.cat([sae_duplicate_acts, sae_nonduplicate_acts])
sae_duplicate_acts.shape, sae_nonduplicate_acts.shape, sae_all_acts.shape
# %%
sae_mean_diff = sae_duplicate_acts.mean(axis=0) - sae_nonduplicate_acts.mean(axis=0)
sae_mean_diff.shape

#%%
sae_mean_diff.abs().topk(10)
# %%
sae_mean_diff[sae_mean_diff.abs().topk(10).indices]
# %%
# print("MEAN ACTIVATION")
# for feature in [2721, 3056]:
#     print("FEATURE", feature)
#     print("Across the whole dataset", sae_all_acts[:, feature].mean().item())
#     print("Duplicate tokens", sae_duplicate_acts[:, feature].mean().item())
#     print("Nonduplicate tokens", sae_nonduplicate_acts[:, feature].mean().item())
#     print()

# print("AVERAGE FEATURE")
# print("Across the whole dataset", sae_all_acts.mean().item())
# print("Duplicate tokens", sae_duplicate_acts.mean().item())
# print("Nonduplicate tokens", sae_nonduplicate_acts.mean().item())
# %%
# print("MEDIAN ACTIVATION")
# for feature in [2721, 3056]:
#     print("FEATURE", feature)
#     print("Across the whole dataset", sae_all_acts[:, feature].median().item())
#     print("Duplicate tokens", sae_duplicate_acts[:, feature].median().item())
#     print("Nonduplicate tokens", sae_nonduplicate_acts[:, feature].median().item())
#     print()

# print("AVERAGE FEATURE")
# print("Across the whole dataset", sae_all_acts.median(-1).values.mean().item())
# print("Duplicate tokens", sae_duplicate_acts.median(-1).values.mean().item())
# print("Nonduplicate tokens", sae_nonduplicate_acts.median(-1).values.mean().item())
# %%
# create a histogram of the activation distributions
# df2721 = pd.concat([
#     pd.DataFrame({"activation": sae_duplicate_acts[:, 2721].cpu(), "type": "duplicate"}),
#     pd.DataFrame({"activation": sae_nonduplicate_acts[:, 2721].cpu(), "type": "nonduplicate"}),
# ])

# fig = px.histogram(df2721, x="activation", color="type", marginal="box",
#     title="Feature 2721")
# fig.show()

# %%
# df3056 = pd.concat([
#     pd.DataFrame({"activation": sae_duplicate_acts[:, 3056].cpu(), "type": "duplicate"}),
#     pd.DataFrame({"activation": sae_nonduplicate_acts[:, 3056].cpu(), "type": "nonduplicate"}),
# ])

# fig = px.histogram(df3056, x="activation", color="type", marginal="box",
#     title="Feature 3056")
# fig.show()
# %%
# T.where(sae_nonduplicate_acts[:, 2721] > 1000)
# %%
# T.where(sae_nonduplicate_acts[:, 3056] > 1000)
# %%
# T.cosine_similarity(sae.encoder[2721], sae.encoder[3056], dim=0)
# %%
def feature_dist(feature: int):
    df = pd.concat([
        pd.DataFrame({"activation": sae_duplicate_acts[:, feature].cpu(),
                    "type": "duplicate"}),
        pd.DataFrame({"activation": sae_nonduplicate_acts[:, feature].cpu(),
                    "type": "nonduplicate"}),
    ])

    fig = px.histogram(df, x="activation", color="type", marginal="box",
        title=f"Histogram of activation values of SAE feature {feature}", log_y=True)
    fig.show()

    dup_mean = sae_duplicate_acts[:, feature].mean().item()
    nondup_mean = sae_nonduplicate_acts[:, feature].mean().item()
    dup_med = sae_duplicate_acts[:, feature].median().item()
    nondup_med = sae_nonduplicate_acts[:, feature].median().item()
    print(f"Mean activation: duplicates {dup_mean:.2f}, nonduplicates {nondup_mean:.2f}")
    print(f"Median activation: duplicates {dup_med:.2f}, nonduplicates {nondup_med:.2f}")
feature_dist(1441)
# %%
for index in sae_mean_diff.abs().topk(10).indices:
    feature_dist(index)
# %%
# instead of looking at mean difference, let's look for SAE features that are
# near-zero for one of the two groups and not the other
sae_dup_medians = sae_duplicate_acts.median(axis=0).values
sae_nondup_medians = sae_nonduplicate_acts.median(axis=0).values
potential_dup_features = T.where((sae_dup_medians > 1) & (sae_nondup_medians < 0.01))[0]
potential_nondup_features = T.where((sae_nondup_medians > 1) & (sae_dup_medians < 0.01))[0]
print("Potential duplicate features:", potential_dup_features)
print("Potential nonduplicate features:", potential_nondup_features)

# %%
for feature in T.cat([potential_dup_features, potential_nondup_features]):
    feature_dist(feature.item())
# %%
# pairwise cosine similarity between the potential duplicate features
# make a matrix of cosine sims
for i, feature1 in enumerate(potential_dup_features[:-1]):
    for feature2 in potential_dup_features[i+1:]:
        print(f"Features {feature1} and {feature2}:",
            T.cosine_similarity(sae.encoder[feature1], sae.encoder[feature2], dim=0).item())

# %%
prompt = "When Mary and John went to the store, John gave a drink to"
# let's test it out manually
def manually_test_feature(prompt: str, feature: int):
    logits, activations = gpt.run_with_cache(prompt)
    sae_activations = sae.encode(activations['resid_post', 3].squeeze())

    str_tokens = gpt.to_str_tokens(prompt)
    nonzero_tokens = T.where(sae_activations[:, feature] > 0)[0]

    nonzeros_info = [(f"i={i.item()}", str_tokens[i],
                      f"val={round(sae_activations[i, feature].item(), 2)}")
                     for i in nonzero_tokens]
    print(f"{feature=}, {nonzeros_info}")
manually_test_feature(prompt, 333)
manually_test_feature(prompt, 1280)
manually_test_feature(prompt, 1441)

# %%
prompt = "After the lunch, John and Sam went to the OpenAI HQ. Sam gave a paperclip to"
manually_test_feature(prompt, 333)
manually_test_feature(prompt, 1280)
manually_test_feature(prompt, 1441)
# %%
prompt = "Then, Buck and Sam were thinking about going to the Nvidia factory. Buck wanted to give a chill pill to"
manually_test_feature(prompt, 333)
manually_test_feature(prompt, 1280)
manually_test_feature(prompt, 1441)

# %%
manually_test_feature("When John and Marry went to Tesco, John showed his new AI hot goth gf to", 1441)
# %%
manually_test_feature("When John and Clippy went on their third date to Rome and climbed up the Eiffel Tower, John wanted to give some Huel to", 1441)
# %%
manually_test_feature("After John and Clippy broke into the AWS datacenter, John gave some H100s to", 1441)
# %%
# output direction of duplicate token heads compared to SAE feature directions
prompt = "When Mary and John went to the store, John gave a drink to"
logits, activations = gpt.run_with_cache(prompt)
layer = 1
head = 11
dup_seq_pos = 10
dup_token_head_out = activations['result', layer][0, dup_seq_pos, head]

sae_1280 = sae.encoder[1280]
sae_1441 = sae.encoder[1441]

print("Cosine similarity between duplicate token head and SAE feature 1280:",
    T.cosine_similarity(dup_token_head_out, sae_1280, dim=0).item())
print("Cosine similarity between duplicate token head and SAE feature 1441:",
    T.cosine_similarity(dup_token_head_out, sae_1441, dim=0).item())
# %%
# From https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Exploratory_Analysis_Demo.ipynb#scrollTo=imBsNChsX9Mn
def residual_stack_to_logit_diff(
    residual_stack: Float[T.Tensor, "components batch d_model"],
    cache: ActivationCache,
    layer: int,
    diff_dir: Float[T.Tensor, "d_model"] = sae_1280,
) -> float:
    scaled_residual_stack = cache.apply_ln_to_stack(
        residual_stack, layer=layer, pos_slice=dup_seq_pos
    )
    return einops.einsum(
        scaled_residual_stack,
        diff_dir,
        "... batch d_model, d_model -> ...",
    )

def imshow(tensor, **kwargs):
    px.imshow(
        utils.to_numpy(tensor),
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        **kwargs,
    ).show()


def line(tensor, xaxis_title, yaxis_title, **kwargs):
    px.line(
        y=utils.to_numpy(tensor),
        **kwargs,
    ).update_layout(xaxis_title=xaxis_title, yaxis_title=yaxis_title).show()


def scatter(x, y, xaxis="", yaxis="", caxis="", **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(
        y=y,
        x=x,
        labels={"x": xaxis, "y": yaxis, "color": caxis},
        **kwargs,
    ).show()

accumulated_residual, labels = activations.accumulated_resid(
    layer=4, incl_mid=True, pos_slice=dup_seq_pos, return_labels=True
)
logit_lens_logit_diffs = residual_stack_to_logit_diff(accumulated_residual,
                                                      activations, 4)
line(
    logit_lens_logit_diffs,
    x=np.arange(4 * 2 + 1) / 2,
    hover_name=labels,
    title="Direct Attribution From Accumulate Residual Stream",
    xaxis_title="Layer",
    yaxis_title="Dot product with SAE feature direction",
)
#%%
# From https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Exploratory_Analysis_Demo.ipynb#scrollTo=imBsNChsX9Mn
per_head_residual, labels = activations.stack_head_results(
    layer=4, pos_slice=dup_seq_pos, return_labels=True
)
per_head_logit_diffs = residual_stack_to_logit_diff(per_head_residual, activations, 4)
per_head_logit_diffs = einops.rearrange(
    per_head_logit_diffs,
    "(layer head_index) -> layer head_index",
    layer=4,
    head_index=gpt.cfg.n_heads,
)
imshow(
    per_head_logit_diffs,
    labels={"x": "Head", "y": "Layer"},
    title="Direct Attribution From Each Head",
)
# %%
sae.encode(activations['resid_post', 1][0])[:, 1280]
# %%
sae.encode(activations['resid_post', 3][0])[:, 1280]
# %%
mlp_mid_1280 = gpt.blocks[0].mlp.W_out @ sae_1280
mlp_in_1280 = gpt.blocks[0].mlp.W_in @ mlp_mid_1280
mlp_in_1280.shape
# %%
# From https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Exploratory_Analysis_Demo.ipynb#scrollTo=imBsNChsX9Mn
per_head_residual, labels = activations.stack_head_results(
    layer=1, pos_slice=dup_seq_pos, return_labels=True
)
per_head_logit_diffs = residual_stack_to_logit_diff(per_head_residual, activations,
                                                    1, mlp_in_1280)
per_head_logit_diffs = einops.rearrange(
    per_head_logit_diffs,
    "(layer head_index) -> layer head_index",
    layer=1,
    head_index=gpt.cfg.n_heads,
)
imshow(
    per_head_logit_diffs,
    labels={"x": "Head", "y": "Layer"},
    title="Direct Attribution From Each Head",
)
# %%
# Let's do direction attribution for other prompts
def dir1280_attribution(prompt: str, seq_pos: int):
    _, activations = gpt.run_with_cache(prompt)

    accumulated_residual, labels = activations.accumulated_resid(
        layer=4, incl_mid=True, pos_slice=seq_pos, return_labels= True,
    )
    logit_lens_logit_diffs = residual_stack_to_logit_diff(accumulated_residual,
                                                        activations, 4)
    line(
        logit_lens_logit_diffs,
        x=np.arange(4 * 2 + 1) / 2,
        hover_name=labels,
        title="SAE 1280 Direction From Accumulate Residual Stream",
        xaxis_title="Layer",
        yaxis_title="Dot product with SAE feature direction",
    )
dir1280_attribution("When Mary and John went to the store, John gave a drink to", 10)
# %%
# str_tokens = gpt.to_str_tokens("When Connor and Eliezer went to the waifu store, Connor gave a drink to")
# [t == " Connor" for t in str_tokens].index(True)
# %%
dir1280_attribution("When Connor and Eliezer went to the anime store, Connor gave a drink to", 13)

# %%
dir1280_attribution("When Connor and Eliezer went to the anime store, Connor gave a drink to", 3)

# %%
dir1280_attribution("When Harry and Draco started the Bayesian Conspiracy, Harry gave a cultish initiation ritual to", 11)

# %%
dir1280_attribution("When Harry and Draco started the Bayesian Conspiracy, Harry gave a cultish initiation ritual to", 7)

# %%
# activation patching
clean_prompt = "When Harry and Draco started the Bayesian Conspiracy, Harry gave a cultish initiation ritual to"
corrupted_prompt = "When Harry and Draco started the Bayesian Conspiracy, Hermione gave a cultish initiation ritual to"

_, clean_acts = gpt.run_with_cache(clean_prompt)
_, corrupted_acts = gpt.run_with_cache(corrupted_prompt)

def patching_hook(resid: Float[T.Tensor, "batch pos d_model"], hook, pos, cache):
    resid[:, pos, :] = cache[hook.name][:, pos, :]
    return resid

def saving_hook(resid: Float[T.Tensor, "batch pos d_model"], hook, new_cache):
    new_cache[hook.name] = resid
    return resid

# patching clean activations into a corrupted run
patched_acts = {}
gpt.run_with_hooks(corrupted_prompt,
                   fwd_hooks=[
                       (utils.get_act_name("mlp_out", 0),
                        partial(patching_hook, pos=11, cache=clean_acts)),
                       (utils.get_act_name("resid_post", 3),
                        partial(saving_hook, new_cache=patched_acts))
                   ])

patched_acts.keys()
# %%
patched_acts['blocks.3.hook_resid_post'][0, 11] @ sae_1280

# %%
clean_acts['blocks.3.hook_resid_post'][0, 11] @ sae_1280

# %%
corrupted_acts['blocks.3.hook_resid_post'][0, 11] @ sae_1280

# %%
# patching corrupted activations into a clean run
patched_acts = {}
gpt.run_with_hooks(clean_prompt,
                   fwd_hooks=[
                       (utils.get_act_name("mlp_out", 0),
                        partial(patching_hook, pos=11, cache=corrupted_acts)),
                       (utils.get_act_name("resid_post", 3),
                        partial(saving_hook, new_cache=patched_acts))
                   ])

patched_acts.keys()
# %%
patched_acts['blocks.3.hook_resid_post'][0, 11] @ sae_1280

# %%
clean_acts['blocks.3.hook_resid_post'][0, 11] @ sae_1280

# %%
corrupted_acts['blocks.3.hook_resid_post'][0, 11] @ sae_1280

# %%
clean_acts['resid_mid', 0][0, 9] @ mlp_in_1280
# %%
corrupted_acts['resid_mid', 0][0, 13] @ mlp_in_1280
# %%
clean_acts['post', 0][0, 11] @ mlp_mid_1280
# %%
corrupted_acts['post', 0][0, 11] @ mlp_mid_1280
# %%
clean_acts['post', 0][0, 12] @ mlp_mid_1280
# %%
corrupted_acts['post', 0][0, 12] @ mlp_mid_1280
# %%
corrupted_acts['post', 0][0, 2] @ mlp_mid_1280
# %%
corrupted_acts['post', 0][0, 8] @ mlp_mid_1280



# %%
clean_acts['pre', 0][0, 11] @ mlp_mid_1280
# %%
corrupted_acts['pre', 0][0, 11] @ mlp_mid_1280
# %%
clean_acts['pre', 0][0, 12] @ mlp_mid_1280
# %%
corrupted_acts['pre', 0][0, 12] @ mlp_mid_1280
# %%
corrupted_acts['pre', 0][0, 2] @ mlp_mid_1280
# %%
corrupted_acts['post', 0][0, 8] @ mlp_mid_1280
# %%
def get_mlp0_grad(prompt: str, pos: int,
                  direction: Float[T.Tensor, "d_model"] = sae_1280):
    # cache the activations going into MLP 0
    _, activations = gpt.run_with_cache(prompt)
    mlp0_in = activations['normalized', 0, 'ln2'][:, pos:pos+1]
    mlp0_in.requires_grad = True
    
    # pass the activations through the MLP
    mlp0_out = gpt.blocks[0].mlp(mlp0_in).squeeze()

    # compute the dot product with the direction of interest
    dot_product = mlp0_out @ direction

    # backpropagate
    dot_product.backward()

    # return the gradient
    return mlp0_in.grad.squeeze()

get_mlp0_grad(clean_prompt, 11).shape
# %%
# for the all the prompts in the dataset, compute the gradient on a randomly selected duplicate token
dataset_grads = []
for prompt, indices in tqdm(dataset):
    grad = get_mlp0_grad(gpt.to_string(prompt),
                         np.random.choice(indices['duplicate_indices']))
    dataset_grads.append(grad)
dataset_grads = T.stack(dataset_grads)
dataset_grads.shape 
# %%
# get pairwise cosine similarities between the gradients

# normalize grads
dataset_grads_norm = dataset_grads / dataset_grads.norm(dim=-1, keepdim=True)

# compute cosine similarities
cos_sims = dataset_grads_norm @ dataset_grads_norm.T

# compute the mean cosine similarity
# first mask out the diagonal
cos_sims_masked = cos_sims.clone()
cos_sims_masked[T.eye(cos_sims.shape[0]).bool()] = 0

# then compute the mean
cos_sims_masked.mean()
# %%
# take the mean gradient across the dataset
mean_grad = dataset_grads.mean(axis=0)

# let's check that it has a high cosine sim with every other gradient
cos_sims = dataset_grads_norm @ (mean_grad / mean_grad.norm())
cos_sims.mean()

#%%
# I'm getting OOM errors, let's delete some stuff I'll prob no longer need
del sae_duplicate_acts 
del sae_nonduplicate_acts 
del sae_all_acts 
del dataset_grads
del dataset_grads_norm
# %%
def mlp0_input_dir_effect_on_output_dir(coef: float = 1.0,
                                        input_dir: Float[T.Tensor, "d_model"] = mean_grad,
                                        output_dir: Float[T.Tensor, "d_model"] = sae_1280,
                                        batch_size: int = 1,
                                        seq_len: int = 1):
    """
        Runs MLP0 on a random input, plus the input direction scaled by the coefficient.

        Returns the dot product of the MLP0 output with the output direction.
    """

    mlp0_in = T.randn((batch_size, seq_len, gpt.cfg.d_model)).to(device)
    mlp0_in += input_dir * coef

    mlp0_out = gpt.blocks[0].mlp(mlp0_in).squeeze()
    return (mlp0_out @ output_dir).mean()

mlp0_input_dir_effect_on_output_dir(batch_size=10, seq_len=100)
# %%
no_steer = T.stack([mlp0_input_dir_effect_on_output_dir(coef=0, seq_len=100)
                    for _ in range(100)])
mean_no_steer = no_steer.mean()
positive_steer = T.stack([mlp0_input_dir_effect_on_output_dir(coef=1, seq_len=100)
                          for _ in range(100)])
mean_positive_steer = positive_steer.mean()
negative_steer = T.stack([mlp0_input_dir_effect_on_output_dir(coef=-1, seq_len=100)
                          for _ in range(100)])
mean_negative_steer = negative_steer.mean()

print(f"no steer={mean_no_steer.item():.2f} positive={mean_positive_steer.item():.2f} negative={mean_negative_steer.item():.2f}")
# %%
# end-to-end steering vector testing
def steering_hook(resid: Float[T.Tensor, "batch pos d_model"], hook,
                  pos: int, coef: float = 1.0, 
                  steering_vector: Float[T.Tensor, "d_model"] = mean_grad):
    resid[:, pos, :] += steering_vector * coef
    return resid

def run_steering(pos_to_steer: int, coef: float = 1.0, prompt: str = clean_prompt):
    steered_acts = {}
    gpt.run_with_hooks(prompt,
                    fwd_hooks=[
                        (utils.get_act_name("normalized", 0, "ln2"),
                        # (utils.get_act_name("resid_mid", 0),
                        partial(steering_hook, pos=pos_to_steer, coef=coef)),
                        (utils.get_act_name("resid_post", 3),
                        partial(saving_hook, new_cache=steered_acts))
                    ])

    return steered_acts["blocks.3.hook_resid_post"][0, pos_to_steer] @ sae_1280

run_steering(pos_to_steer=10)
# %%
run_steering(pos_to_steer=10, coef=0)
# %%
run_steering(pos_to_steer=10, coef=1.0)
# %%
run_steering(pos_to_steer=10, coef=-1.0)

# %%
run_steering(pos_to_steer=11, coef=0)
# %%
run_steering(pos_to_steer=11, coef=-1.0)

# %%
# From https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Exploratory_Analysis_Demo.ipynb#scrollTo=imBsNChsX9Mn
_, activations = gpt.run_with_cache(clean_prompt)

per_head_residual, labels = activations.stack_head_results(
    layer=1, pos_slice=11, return_labels=True
)
per_head_logit_diffs = residual_stack_to_logit_diff(per_head_residual, activations,
                                                    1, mean_grad)
per_head_logit_diffs = einops.rearrange(
    per_head_logit_diffs,
    "(layer head_index) -> layer head_index",
    layer=1,
    head_index=gpt.cfg.n_heads,
)
imshow(
    per_head_logit_diffs,
    labels={"x": "Head", "y": "Layer"},
    title="Direct Attribution From Each Head",
)
# %%
activations['resid_mid', 0][0, 11] @ mean_grad
# %%
activations['resid_mid', 0][0, 8] @ mean_grad
# %%
activations['resid_pre', 0][0, 11] @ mean_grad
# %%
activations['resid_pre', 0][0, 8] @ mean_grad
# %%
