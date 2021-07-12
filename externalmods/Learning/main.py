import sys, os
from . import io, prep, train

if __name__ == "__main__":
    # python -m modeling.main [build, update] [all, mem_bin, memory, wallclock]
    # default args: [build] [all] # builds and trains all models using all data
    args = sys.argv
    options = ["build", "update"]
    models = ["all", "mem_bin", "memory", "wallclock"]
    if len(args) > 2:
        opt, mod = args[1], args[2]
    elif len(args) == 2:
        opt, mod = args[1], "all"
    else:
        opt, mod = "build", "all"
    if opt not in options:
        print(f"Invalid option arg: {opt}")
        print(f"Options: {options}")
        opt = "build"
    if mod not in models:
        print(f"Invalid model arg: {mod}")
        print(f"Mods: {models}")
        mod = "all"
    print("flags:", [opt, mod])
    # AWS Batch job simplifies setting of default Env vars
    bucket_mod = os.environ.get("S3MOD", "calcloud-modeling-sb") # where to pull and store metadata
    timestamp = os.environ.get("TIMESTAMP", "now")  # results saved to timestamped directory (s3)
    verbose = os.environ.get("VERBOSE", 0) # print everything to stdout (set=1 for debug)
    cross_val = os.environ.get("KFOLD", None) # 'only', 'skip', or None
    src = os.environ.get("DATASOURCE", "ddb")
    table_name = os.environ.get("DDBTABLE", "calcloud-model-sb")
    p_key = os.environ.get("PKEY", "ipst") # primary key (dynamodb)
    attr = os.environ.get("ATTR", None) # retrieve subset from dynamodb
    data_path = io.get_paths(timestamp)
    home = os.path.join(os.getcwd(), data_path)
    prefix = f"{data_path}/data"
    os.makedirs(prefix, exist_ok=True)
    os.chdir(prefix)
    df = prep.preprocess(bucket_mod, prefix, src, table_name, p_key, attr)
    os.chdir(home)
    train.train_models(df, bucket_mod, data_path, opt, mod, verbose, cross_val)
    io.zip_models("./models", zipname="models.zip")
    io.s3_upload(["models.zip"], bucket_mod, f"{data_path}/models")