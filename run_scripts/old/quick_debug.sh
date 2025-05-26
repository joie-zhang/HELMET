CMD1="python eval.py --config configs/debug.yaml --model_name_or_path /scratch/gpfs/DANQIC/models/Llama-3.1-8B-Instruct --tokenizer_name /scratch/gpfs/DANQIC/models/Llama-3.1-8B-Instruct --minference snapkv --output_dir outputs/debug/PATCH64_2_outputsPATCH__sp0_pf32768_tg__snapkv --no_torch_compile --minference_sparsity 0 --minference_window_size 64 --minference_chunk_prefilling 32768 --minference_chunking_patch --overwrite"

CMD2="python eval.py --config configs/debug.yaml --model_name_or_path /scratch/gpfs/DANQIC/models/Llama-3.1-8B-Instruct --tokenizer_name /scratch/gpfs/DANQIC/models/Llama-3.1-8B-Instruct --minference snapkv --output_dir outputs/debug/PATCH64_2_outputsPATCH__sp0_pf32768_tg__snapkv --no_torch_compile --minference_sparsity 0 --minference_window_size 64 --minference_chunk_prefilling 32768 --overwrite"

CMD3="python eval.py --config configs/debug.yaml --model_name_or_path /scratch/gpfs/DANQIC/models/Llama-3.1-8B-Instruct --tokenizer_name /scratch/gpfs/DANQIC/models/Llama-3.1-8B-Instruct --minference snapkv --output_dir outputs/debug/LOCAL64_2_outputs_sp_tg__snapkv --no_torch_compile --minference_sparsity 0 --minference_window_size 64 --minference_chunk_prefilling 131072 --overwrite"

$CMD1
$CMD2
$CMD3