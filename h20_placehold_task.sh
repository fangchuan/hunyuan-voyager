arena --loglevel info submit pytorch  \
    --toleration 'all' \
    --namespace mri \
    --name zhenqing-debug-voyager \
    --gpus 1 \
    --workers 1 \
    --cpu 24 \
    --memory 100Gi \
    --hostNetwork true \
    --data-dir /data-nas:/data-nas \
    --data-dir /data-high-nas:/data-high-nas \
    --image registry.qunhequnhe.com/mri/trellis:metaquery \
    --image-pull-policy  IfNotPresent \
    --working-dir=/workspace \
    --sync-mode git \
    --sync-source https://gitlab.qunhequnhe.com/zhenqing/Spatial3DV.git \
    --sync-branch develop \
    --clean-task-policy None \
    --shell "bash" \
    "env;
cd /workspace/code/Spatial3DV;
echo \$PWD;

pip config set global.index-url https://mirrors.aliyun.com/pypi/simple;
pip install jaxtyping  omegaconf typeguard shapely pyexr loguru colorama torchmetrics[image]==0.11.4 openai-clip agentscope; 
pip install diffusers==0.31.0 tensorboard==2.19.0 transformers==4.39.3 deepspeed==0.15.1 peft==0.13.2 pyexr==0.5.0 decord;

export DATASET_PATH=/data-high-nas/data/dataset/qunhe/Spatial3D/v7_128/;
export OUTPUT_FOLDER=/data-nas/data/experiments/zhenqing/trellis/debug_output;

export DINO_PRETRAINED_MODEL_PATH=/data-nas/models/dinov2/dinov2_vitl14_reg4_pretrain.pth;
mkdir -p /root/.cache/torch/hub/checkpoints;
cp \$DINO_PRETRAINED_MODEL_PATH /root/.cache/torch/hub/checkpoints/;

sleep 344h
"
