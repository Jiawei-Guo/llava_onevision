# export OMP_NUM_THREADS=8
# export NCCL_IB_DISABLE=0
# export NCCL_IB_GID_INDEX=3
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_DEBUG=INFO

LLM_VERSION="/xpfs/public/models/hf_models/Meta-Llama-3-8B-Instruct/"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="openai/clip-vit-large-patch14-336"
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
# llama3 transformers 4.40.0.dev0
export HF_HOME="/xpfs/public/research/jiawei/cache"
# /ML-A100/public/run/research/gjw/LLaVA/scripts/v1_5/fintune_test_7b.sh

WANDB_API_KEY='37128389d5735eafeee374833b0b2391aeffe924'
wandb login --relogin $WANDB_API_KEY

PROMPT_VERSION=plain

BASE_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-pretrain_blip558k_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"
# ACCELERATE_CPU_AFFINITY=1 
torchrun --nproc_per_node $GPUS_PER_NODE \
 --master_addr $MASTER_ADDR \
 --node_rank $NODE_RANK \
 --master_port $MASTER_PORT \
 --nnodes $NNODES \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path /xpfs/public/research/hf_home/datasets/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder /xpfs/public/research/hf_home/datasets/LLaVA-Pretrain/ \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_tunable_parts="mm_mlp_adapter" \
    --mm_vision_select_layer -2 \
    --mm_projector_type mlp2x_gelu \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /xpfs/public/research/jiawei/LLaVA-NeXT/ckpt/pretrain_llama3 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 50000 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $BASE_RUN_NAME \
    --attn_implementation sdpa

# You can delete the sdpa attn_implementation if you want to use flash attn