import os
import json



def check_dir(d):
    if not os.path.exists(d):
        print(f"Directory {d} does not exist. Exit.")
        exit(1)

def check_files(files):
    for f in files:
        if f is not None and not os.path.exists(f):
            print(f"File {f} does not exist. Exit.")
            exit(1)

def ensure_dir(d, verbose=True):
    if not os.path.exists(d):
        if verbose:
            print(f"Directory {d} do not exist; creating...")
        os.makedirs(d)

def save_config(config, path, verbose=True):
    with open(path, 'w') as outfile:
        json.dump(config, outfile, indent=2)
    if verbose:
        print(f"Config saved to file {path}")
    return config

def load_config(path, verbose=True):
    with open(path) as f:
        config = json.load(f)
    if verbose:
        print(f"Config loaded from file {path}")
    return config

def print_config(config):
    info = "Running with the following configs:\n"
    for k, v in config.items():
        info += f"\t{k} : {v}\n"
    print("\n" + info + "\n")