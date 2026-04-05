#!/bin/python3

import argparse
import pathlib
import pandas as pd
# from template import Template as yaml_template
from utils import get_run_name, GPU_MAP, TOKEN_LENGTH_MAP

def write_yaml_files(target_dir, \
    file_content, \
    inference_engine, \
    gpu, \
    num_gpu, \
    csv_file_name):

    file_name = f"{inference_engine}_{gpu}x{num_gpu}_{csv_file_name}.yaml"

    with open(f"{target_dir}/{file_name}", "w") as f:
        f.write(file_content)

def main(experiments_csv, yaml_target_dir, inference_engine):

    if inference_engine=="sglang":
        from template_sglang_loop import Template as yaml_template
    elif inference_engine=="vllm":
        #from template_vllm_loop import Template as yaml_template
        print("Inference engine: '", inference_engine, "' under construction")
        raise SystemExit(1)
    else:
        print("Inference engine: '", inference_engine, "' not supported")
        raise SystemExit(1)


    experiments_csv_clean = experiments_csv

    pathlib.Path(yaml_target_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(experiments_csv)
    ### Need change logic of this as not getting every row anymore
    df["yaml"] = df.apply(lambda row: yaml_template().get(model_name=row.model_name, \
                                tensor_parallel_size=row.num_gpu, \
                                dataset=row.dataset, \
                                target_input_tokens=TOKEN_LENGTH_MAP[row.target_input_tokens], \
                                target_output_tokens=TOKEN_LENGTH_MAP[row.target_output_tokens], \
                                num_samples=row.num_samples, \
                                batch_size=row.batch_size, \
                                num_gpu=row.num_gpu, \
                                gpu_product=GPU_MAP[row.gpu]), axis=1)
   

    ### Need to change logic here as not producing a file for every row here
    df.apply(lambda row: write_yaml_files(target_dir=yaml_target_dir, \
                                            file_content=row.yaml, \
                                            inference_engine=inference_engine, \
                                            gpu=row.gpu, \
                                            num_gpu=row.num_gpu, \
                                            csv_file_name=experiments_csv_clean), axis=1)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, required=True, help="Path to experiments CSV file")
    parser.add_argument("--target_dir", type=str, required=True, help="Target directory to save generated YAML files")
    parser.add_argument("--inference_engine", type=str, required=True, help="inference engine")
    args = parser.parse_args()

    main(args.csv_file, args.target_dir, args.inference_engine)
