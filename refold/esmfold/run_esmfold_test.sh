export CUDA_VISIBLE_DEVICES=6,7
# python refold/esmfold/run_esmfold_distributed_2.py \
# --name test_esmfold \
# --sequences examples/refold/esmfold/input/all_sequences.json \
# --output_dir examples/refold/esmfold/output \
# --esmfold_model_dir refold/esmfold/weights \
# --verbose_gpu


torchrun \
  --nproc_per_node 2 \
  --nnodes 1 \
  --node_rank 0 \
  --master_addr localhost \
  --master_port 29500 \
  refold/esmfold/run_esmfold_distributed.py \
  --name test_esmfold \
  --sequences examples/refold/esmfold/input/all_sequences.json \
  --output_dir examples/refold/esmfold/output \
  --esmfold_model_dir refold/esmfold/weights \
  --verbose_gpu

