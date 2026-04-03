apiVersion: batch/v1
kind: Job
metadata:
  generateName: sglang-moe-cap-
  labels:
    kueue.x-k8s.io/queue-name:  eidf230ns-user-queue
spec:
  completions: 3
  backoffLimit: 0
  ttlSecondsAfterFinished: 1800
  template:
    metadata:
      name: job-sglang-moe-cap
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

            run_name=james_test
            timestamp=23032026_1

            ## On first pass touch a file          
            ## Place 0 in file
            ## This 0 represents the element in the array of row id's of the experiments csv that was last tested
            ## Each subsequent access goes to the next row in the list

            declare -i iter

            echo "check mnt location"

            ls /mnt/ceph

            mkdir -p /mnt/ceph/jr_iter_test3

            iter_path=/mnt/ceph/jr_iter_test3

            echo "iter path:"
            echo $iter_path

            # if [[ ! -e iteration_file ]]; then touch iteration_file; iter=0; echo $iter &> iteration_file; else iter=$(cat iteration_file); fi

            if [[ ! -e $iter_path/iteration_file ]]; then touch $iter_path/iteration_file; iter=0; echo $iter &> $iter_path/iteration_file; else iter=$(cat $iter_path/iteration_file); fi

            ## Array of rows id's to be accessed in this test
            ## defined during template generation

            rowIds=('15' '16' '17')


            echo "test iteration file created"
            cat $iter_path/iteration_file


            ## Implicit start of do-while loop

            ### access next element of the array of row ids
            
            row_number="${rowIds[iter]}"

            echo "iter:"
            echo $iter
            echo "row number:"
            echo $row_number
            ### Load experiments.csv as table (in python)


            wget https://github.com/JPRichings/TEASBench/raw/refs/heads/main/MoE-Benchmark/direct-test-scripts/parameter.py

            wget https://github.com/JPRichings/TEASBench/raw/refs/heads/main/MoE-Benchmark/direct-test-scripts/data/experiments.csv

            ### select correct row and copy elements in the temp variables

            tmp_model_name=$(python3 parameter.py --csv_file=experiments.csv --parameter_name="model_name" --experiment_id=$row_number)
            tmp_tensor_parallel_size=$(python3 parameter.py --csv_file=experiments.csv --parameter_name="num_gpu" --experiment_id=$row_number) 
            tmp_dataset=$(python3 parameter.py --csv_file=experiments.csv --parameter_name="dataset" --experiment_id=$row_number)
            tmp_target_input_tokens=4000 #$(python3 parameter.py --csv_file=experiments.csv --parameter_name="target_input_tokens" --experiment_id=$row_number)
            tmp_target_output_tokens=1000 #$(python3 parameter.py --csv_file=experiments.csv --parameter_name="target_output_tokens" --experiment_id=$row_number)
            tmp_num_samples=$(python3 parameter.py --csv_file=experiments.csv --parameter_name="num_samples" --experiment_id=$row_number)
            tmp_batch_size=$(python3 parameter.py --csv_file=experiments.csv --parameter_name="batch_size" --experiment_id=$row_number)


            echo "model name" $tmp_model_name
            echo "tensor parrallel size" $tmp_tensor_parallel_size
            echo "dataset" $tmp_dataset
            echo "target_input_tokens" $tmp_target_input_tokens
            echo "target_output_tokens" $tmp_target_output_tokens
            echo "num_samples" $tmp_num_samples
            echo "batch_size" $tmp_batch_size

            # Start Sglang server
            python -m moe_cap.systems.sglang --model-path $tmp_model_name --port 30000 --expert-distribution-recorder-mode stat --tp-size $tmp_tensor_parallel_size &> /dev/shm/${run_name}_${timestamp}.server_log & # REPLACE run_name and timestamp
            SERVER_PID=$!

            # Wait until the /health endpoint returns HTTP 200
            echo "Waiting for SGLang server to be ready..."

            until curl -s -f http://localhost:30000/health > /dev/null; do
              echo -n "."
              sleep 2
            done

            echo "SGLang server is ready!"
            echo "Starting to serve bench (sending http requests)..."
            
            mkdir -p /dev/shm/${run_name}
            python -m moe_cap.runner.openai_api_profile --model_name ${tmp_model_name} --datasets ${tmp_dataset} --input-tokens ${tmp_target_input_tokens} --output-tokens ${tmp_target_output_tokens} --num-samples ${tmp_num_samples} --config-file configs/stub.yaml --api-url http://localhost:30000/v1/completions --backend sglang --ignore-eos --server-batch-size ${tmp_batch_size} --output_dir /dev/shm/${run_name} &> /dev/shm/${run_name}_${timestamp}.client_log # REPLACE run_name timestamp

            echo "Starting to serve bench (sending http requests)... done!"
            echo "Benchmark finished, shutting down server..."

            kill $SERVER_PID
            wait $SERVER_PID
            
            echo "Server stopped. Copying files to pvc..."
            
            mkdir -p /mnt/ceph/tmp/MoE-CAP-outputs
            cp -R /dev/shm/${run_name} /mnt/ceph/tmp/MoE-CAP-outputs/jr_test/
            cp /dev/shm/${run_name}_${timestamp}* /mnt/ceph/tmp/MoE-CAP-outputs/jr_test/ # REPLACE run_name timestamp
            
            echo "Files copied, exiting container"

            ## Update iter file for next iteration

            if [[ iter -le ${#rowIds[@]} ]]; then
              iter+=1
              echo "updated iter:"               
              echo $iter
              echo $iter > $iter_path/iteration_file
              echo "iteration count incremented:"
              cat $iter_path/iteration_file
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
            nvidia.com/gpu: 1 
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
        nvidia.com/gpu.product: NVIDIA-A100-SXM4-80GB
