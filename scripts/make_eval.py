'''Creates eval configs with _slim'''
import json
import glob
import os

os.mkdir("eval_configs")
for fn in glob.glob("configs/*.json"):
    data = json.load(open(fn))
    data["model_dir"] = data["model_dir"] + "_slim"
    out_fn = "eval_configs/" + fn.split("/")[-1]
    print("Created", out_fn)
    json.dump(data, open(out_fn, 'w'))
