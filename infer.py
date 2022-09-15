import argparse
import json
import time
from loguru import logger

import jax
import numpy as np
import optax
from tqdm import tqdm

from mesh_transformer import util
from mesh_transformer.checkpoint import read_ckpt
from mesh_transformer.sampling import nucleaus_sample
from mesh_transformer.transformer_shard import CausalTransformer
import transformers
from smart_open import open

from mesh_transformer.util import clip_by_global_norm 
import datasets
import re

CHUNK_SIZE = 512

TASK_MAP = {
    "translation": {
        "model": "ghomasHudson/gptj_long_contra_pro",
        "dataset": "ghomasHudson/long_contra_pro",
        "output_max_len": CHUNK_SIZE + 10
    },
    "char_id": {
        "model": "ghomasHudson/character_id",
        "dataset": "ghomasHudson/character_id",
        "output_max_len": 5
    },
    "qa": {
        "model": "ghomasHudson/character_id",
        "dataset": "ghomasHudson/hotpotExtendedAno",
        "output_max_len": 30
    },
    "summarization": {
        "model": "ghomasHudson/booksum",
        "dataset": "ghomasHudson/booksum_ds",
        "output_max_len": int(CHUNK_SIZE * 0.75)
    }
}

TASK = "summarization"



def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Config file location")

    args = parser.parse_args()
    return args


def parse_lm_string(s):
    '''Turns a language model string with <|key|> value sections into a dict'''
    sSplit = re.split(r'<\|(.+)\|>', s)
    out_dict = {}
    for i in range(1, len(sSplit), 2):
        out_dict[sSplit[i]] = sSplit[i+1].strip()
    return out_dict


def make_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    was_str = False
    if isinstance(lst, str):
        was_str = True
        lst = lst.split(" ")

    for i in range(0, len(lst), n):
        if was_str:
            yield " ".join(lst[i:i + n])
        else:
            yield lst[i:i + n]


def infer(context, output_length=512):
    '''Produces an inference from a context input'''
    start = time.time()
    tokens = tokenizer.encode(context)
    provided_ctx = len(tokens)
    pad_amount = seq - provided_ctx

    padded_tokens = np.pad(tokens, ((pad_amount, 0),)).astype(np.uint32)
    batched_tokens = np.array([padded_tokens] * total_batch)
    length = np.ones(total_batch, dtype=np.uint32) * len(tokens)

    # generate(ctx,ctx_length, gen_length, sampler_options)
    output = network.generate(
        batched_tokens,
        length,
        output_length,
        {"top_p": np.ones(total_batch) * 0.9, "temp": np.ones(total_batch) * 0.75}
    )
    output = output[1][0][:, :, 0][0]
    text = tokenizer.decode(output)
    #print(f"completion done in {time.time() - start:06}s")
    return text.split("<|endoftext|>", 1)[0]


if __name__ == "__main__":
    args = parse_args()
    params = json.load(open(args.config))

    gradient_accumulation_steps = params.get("gradient_accumulation_steps", 1)
    per_replica_batch = params["per_replica_batch"]
    cores_per_replica = params["cores_per_replica"]

    assert cores_per_replica <= 8

    bucket = params["bucket"]
    model_dir = params["model_dir"]
    layers = params["layers"]
    d_model = params["d_model"]
    n_heads = params["n_heads"]
    n_vocab = params["n_vocab"]
    seq = params["seq"]
    norm = params["norm"]

    params["sampler"] = nucleaus_sample
    opt = optax.chain(
        optax.scale(1 / gradient_accumulation_steps),
        clip_by_global_norm(1),
        optax.scale_by_adam(),
        optax.additive_weight_decay(0),
        optax.scale(-1),
        optax.scale_by_schedule(util.gpt3_schedule(0, 1, 0, 0))
    )

    params["optimizer"] = opt

    start = time.time()
    print(f"jax devices: {jax.device_count()}")
    print(f"jax runtime initialized in {time.time() - start:.06}s")

    mesh_shape = (jax.device_count() // cores_per_replica, cores_per_replica)
    devices = np.array(jax.devices()).reshape(mesh_shape)

    with open(f"gs://{bucket}/{model_dir}/meta.json", "r") as f:
        meta = json.load(f)

    ckpt_step = meta["checkpoints"][-1]
    print(f"using checkpoint {ckpt_step}")

    total_batch = per_replica_batch * jax.device_count() // cores_per_replica
    with jax.experimental.maps.mesh(devices, ('dp', 'mp')):
        # Load network
        network = CausalTransformer(params)

        start = time.time()
        network.state = read_ckpt(network.state, f"gs://{bucket}/{model_dir}/step_{ckpt_step}/", devices.shape[1])
        logger.info(f"network loaded in {time.time() - start:.06}s")

        local_shards = max(jax.local_device_count() // mesh_shape[1], 1)
        del network.state["opt_state"]
        network.state = network.move_xmap(network.state, np.zeros(local_shards))

        tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')

        # ***************************************************
        # Do inference

        # Load dataset
        ds = datasets.load_dataset(TASK_MAP[TASK]["dataset"], use_auth_token=True)
        if "test" in ds:
            ds = ds["test"]
        else:
            ds = ds["validation"]


        # Eval
        preds = []
        trues = []
        for ex in ds:
            start = time.time()
            if TASK == "qa":
                question, text = ex["input"].split("?", 1)
                true_answer = ex["output"][0]
            if TASK == "translation":
                text = ex["translation"]["en"]
                true_answer = ex["translation"]["de"]
            if TASK == "summarization":
                text = ex["input"]
                true_answer = ex["output"]


            intermediate_output = text.strip()
            final_output = ""
            i = 0 
            while len(intermediate_output.split()) > CHUNK_SIZE * 1.2 and i < 20:
                logger.info("")
                logger.info(f"Level {i} - {len(intermediate_output.split())} tokens")
                i += 1
                chunks = list(make_chunks(intermediate_output, CHUNK_SIZE))
                intermediate_output = ""
                if TASK == "qa":
                    chunks = ["<|question|> " + question + "\n<|context|> " + c + "\n<|facts|>" for c in chunks]
                elif TASK == "translation":
                    chunks = ["<|input|> " + c + "\n<|output|>" for c in chunks]
                elif TASK == "summarization":
                    chunks = ["<|input|> " + c + "\n<|output|>" for c in chunks]

                for chunk in tqdm(chunks):
                    output = chunk + infer(chunk, output_length=TASK_MAP[TASK]["output_max_len"])
                    output_d = parse_lm_string(output)
                    if TASK == "qa":
                        intermediate_output += " " + output_d.get("fact", "")
                    elif TASK == "translation":
                        final_output += " " + output_d.get("output", "")
                    if TASK == "summarization":
                        intermediate_output += " " + output_d.get("output", "")

            # Final output
            if TASK == "qa":
                chunk = "<|question|> " + question + "\n<|context|> " + intermediate_output + "\n<|facts|>"
                output = chunk + infer(chunk)
                output_d = parse_lm_string(output)
                pred_answer = output_d.get("answer", "")
            if TASK == "translation":
                #chunk = "<|input|> " + intermediate_output + "\n<|output|>"
                #output = chunk + infer(chunk)
                #output_d = parse_lm_string(output)
                #pred_answer = output_d.get("output", "")
                pred_answer = final_output
            if TASK == "summarization":
                chunk = "<|input|> " + intermediate_output + "\n<|output|>"
                output = chunk + infer(chunk)
                output_d = parse_lm_string(output)
                pred_answer = output_d.get("output", "")

            print(pred_answer[:100])
            print(true_answer[:100])
            preds.append(pred_answer)
            trues.append(true_answer)

            print(f"example done in {time.time() - start:06}s")
            breakpoint()


