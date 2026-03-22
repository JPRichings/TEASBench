#!/bin/python3

from datetime import datetime
from utils import get_run_name

class Template:
    def __init__(self):
        return

    def get(self, \
        model_name: str, \
        tensor_parallel_size: int, \
        dataset: str, \
        target_input_tokens: int, \
        target_output_tokens: int, \
        num_samples: int, \
        batch_size: int, \
        num_gpu: int, \
        gpu_product: str):

        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        model_name_clean=model_name.split("/")[1].replace(".", "-")
        gpu=gpu_product.split("-")[1] 

        run_name = get_run_name(model_name, gpu_product, num_gpu, target_input_tokens, target_output_tokens, batch_size, dataset)
        run_name_lower = run_name.replace("_", "-").lower()
        
        return f"""
apiVersion: batch/v1
kind: Job
metadata:
  generateName: sglang-moe-cap-
  #generateName: sglang-moe-cap-{run_name.replace("_", "-").lower()}-
  labels:
    kueue.x-k8s.io/queue-name:  eidf230ns-user-queue
spec:
  completions: 1
  backoffLimit: 0
  ttlSecondsAfterFinished: 1800
  template:
    metadata:
      name: job-sglang-moe-cap
      #name: job-sglang-moe-cap-{run_name.replace("_", "-").lower()}
    spec:
      containers:
      - name: sglang-server
        image: lmsysorg/sglang:latest
        imagePullPolicy: IfNotPresent
        env:
          - name: SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR
            value: "/dev/shm/sglang_expert_distribution_recorder"
        command: ["/bin/bash", "-c"]
        args:
          - |

            # Install Git 
            apt-get update
            apt-get -y install git

            # Get MoE Cap
            git clone https://github.com/markxio/MoE-CAP.git /dev/shm/MoE-CAP # why is this not the main MoE directory?
            cd /dev/shm/MoE-CAP

            pip install -e .
            pip install gputil

            # Introduce logic to replace variables

            ## On first pass touch a file          
            ## Place 0 in file
            ## This 0 represents the element in the array of row id's of the experiments csv that was last tested
            ## Each subsequent access goes to the next row in the list

            declare -i iter

            if [[ -e iteration_file ]]; then
              touch iteration_file
              iter=0
              echo $iter &> iteration_file
            else
              iter=$(cat iteration_file)
            fi

            ## Array of rows id's to be accessed in this test
            ## defined during template generation

            rowIds = ({row_ids})

            ## Implicit start of do-while loop

            ### access next element of the array of row ids
            
            row_number = "${rowIds[iter]}"

            ### Load experiments.csv as table (in python)

            parameters = $( python row.py )

            ### select correct row and copy elements in the temp variables

            tmp_model_name = "${parameters[0]}"
            tmp_tensor_parallel_size = "${parameters[1]}"
            tmp_dataset = "${parameters[2]}"
            tmp_target_input_tokens = "${parameters[3]}"
            tmp_target_output_tokens = "${parameters[4]}"
            tmp_num_samples = "${parameters[5]}"
            tmp_batch_size = "${parameters[6]}"

            # Start Sglang server
            python -m moe_cap.systems.sglang \\
              --model-path tmp_model_name \\
              --port 30000 \\
              --expert-distribution-recorder-mode stat \\
              --tp-size tmp_tensor_parallel_size \\
              &> /dev/shm/{run_name}_{timestamp}.server_log &
            SERVER_PID=$!

            # Wait until the /health endpoint returns HTTP 200
            echo "Waiting for SGLang server to be ready..."

            until curl -s -f http://localhost:30000/health > /dev/null; do
              echo -n "."
              sleep 2
            done

            echo "SGLang server is ready!"
            echo "Starting to serve bench (sending http requests)..."
            
            mkdir -p /dev/shm/{run_name}
            python -m moe_cap.runner.openai_api_profile \\
              --model_name tmp_model_name \\
              --datasets tmp_dataset \\
              --input-tokens tmp_target_input_tokens \\
              --output-tokens tmp_target_output_tokens \\
              --num-samples tmp_num_samples \\
              --config-file configs/stub.yaml \\
              --api-url http://localhost:30000/v1/completions \\
              --backend sglang \\
              --ignore-eos \\
              --server-batch-size tmp_batch_size \\
              --output_dir /dev/shm/{run_name} \\
              &> /dev/shm/{run_name}_{timestamp}.client_log

            echo "Starting to serve bench (sending http requests)... done!"
            echo "Benchmark finished, shutting down server..."

            kill $SERVER_PID
            wait $SERVER_PID
            
            echo "Server stopped. Copying files to pvc..."
            
            mkdir -p /mnt/ceph/tmp/MoE-CAP-outputs
            cp -R /dev/shm/{run_name} /mnt/ceph/tmp/MoE-CAP-outputs/num_samples_256/
            cp /dev/shm/{run_name}_{timestamp}* /mnt/ceph/tmp/MoE-CAP-outputs/num_samples_256/
            
            echo "Files copied, exiting container"

            ## Update iter file for next iteration

            if [[ iter -le ${#rowIds[@]} ]]; then
              iter+=1
              echo $iter > iteration_file
              echo "iteration count incremented"
            else
              echo "end of iterations"
            fi


        ports:
          - containerPort: 30000 
        resources:
          requests:
            cpu: 10
            memory: '100Gi'
          limits:
            cpu: 10
            memory: '100Gi'
            nvidia.com/gpu: {num_gpu}
        volumeMounts:
          - mountPath: /mnt/ceph
            name: volume
          - mountPath: /dev/shm
            name: dshm
      restartPolicy: Never
      volumes:
        - name: volume
          persistentVolumeClaim:
            claimName: client-ceph-pvc
        - name: dshm
          emptyDir:
            medium: Memory
            sizeLimit: 16Gi
      nodeSelector:
        nvidia.com/gpu.product: {gpu_product}
               """
