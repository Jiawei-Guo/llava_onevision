# export OMP_NUM_THREADS=8
# export NCCL_IB_DISABLE=0
# export NCCL_IB_GID_INDEX=3
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_DEBUG=INFO

LLM_VERSION="/xpfs/public/models/hf_models/llava-onevision-qwen2-7b-mid-stage-a4"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

export GPUS_PER_NODE=${MLP_WORKER_GPU:-${KUBERNETES_CONTAINER_RESOURCE_GPU:-8}}
export NNODES=${MLP_WORKER_NUM:-${WORLD_SIZE:-1}}
export NODE_RANK=${MLP_WORKER_RACK_RANK_INDEX:-${MLP_ROLE_INDEX:-${RANK:-0}}}
export MASTER_ADDR=${MLP_WORKER_0_HOST:-${MASTER_ADDR:-127.0.0.1}}
export MASTER_PORT=${MLP_WORKER_0_PORT:-${MASTER_PORT:-1234}}
export TASK_ID=${MLP_TASK_ID:-$(date "+%Y-%m-%d-%H-%M")}
export WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))

############### Pretrain ################
cd /xpfs/public/research/jiawei/LLaVA-NeXT
source /xpfs/public/research/miniconda3/bin/activate llava_next
# llama3.1 transformers==4.43.1(4.42.3) accelerate==0.33.0 
export HF_HOME="/xpfs/public/research/jiawei/cache"

WANDB_API_KEY='37128389d5735eafeee374833b0b2391aeffe924'
wandb login --relogin $WANDB_API_KEY

# PROMPT_VERSION="llava_llama_3"

# BASE_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-pretrain_blip558k_plain"
# echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

PROMPT_VERSION="qwen_1_5"
MID_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-blip558k_pretrain_plain_la_1_6mix_ft"
echo "MID_RUN_NAME: ${MID_RUN_NAME}"

# ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
torchrun --nproc_per_node $GPUS_PER_NODE \
 --master_addr $MASTER_ADDR \
 --node_rank $NODE_RANK \
 --master_port $MASTER_PORT \
 --nnodes $NNODES \
 llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path /gpfs/public/research/jiawei/visual_web_inst/final_sampled_data_1M.json \
    --image_folder /gpfs/public/research/jiawei/visual_web_inst/image/all_images \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints  "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $MID_RUN_NAME \
    --output_dir "/gpfs/public/research/jiawei/ckpt/sft_llava_qwen_2_1M" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 16384 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True

    # llava/train/train_mem.py \
    # --deepspeed scripts/zero3.json \
    # --model_name_or_path ${LLM_VERSION} \
    # --version ${PROMPT_VERSION} \
    # --data_path /gpfs/public/research/jiawei/visual_web_inst/final_sampled_data_1M.json \
    # --image_folder /gpfs/public/research/jiawei/visual_web_inst/image/all_images \
    # --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    # --mm_vision_tower_lr=2e-6 \
    # --vision_tower ${VISION_MODEL_VERSION} \
    # --mm_projector_type mlp2x_gelu \
    # --mm_vision_select_layer -2 \
    # --mm_use_im_start_end False \
    # --mm_use_im_patch_token False \
    # --group_by_modality_length True \
    # --image_aspect_ratio anyres \
    # --image_grid_pinpoints "[(336, 672), (672, 336), (672, 672), (1008, 336), (336, 1008)]" \
    # --mm_patch_merge_type spatial_unpad \
    # --bf16 True \
    # --run_name $MID_RUN_NAME \
    # --output_dir "/xpfs/public/research/jiawei/LLaVA-NeXT/ckpt/sft_llava_qwen_2_1M" \
    # --num_train_epochs 1 \
    # --per_device_train_batch_size 2 \
    # --per_device_eval_batch_size 2 \
    # --gradient_accumulation_steps 2 \
    # --evaluation_strategy "no" \
    # --save_strategy "steps" \
    # --save_steps 1000 \
    # --save_total_limit 1 \
    # --learning_rate 2e-5 \
    # --weight_decay 0. \
    # --warmup_ratio 0.03 \
    # --lr_scheduler_type "cosine" \
    # --logging_steps 1 \
    # --tf32 True \
    # --model_max_length 32768 \
    # --gradient_checkpointing True \
    # --dataloader_num_workers 16 \
    # --lazy_preprocess True \
    # --report_to wandb \
    # --torch_compile True \
    # --torch_compile_backend "inductor" \
    # --dataloader_drop_last True


# --attn_implementation sdpa
# You can delete the sdpa attn_implementation if you want to use flash attn