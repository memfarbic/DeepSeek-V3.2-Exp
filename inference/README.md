# DeepSeek V3.2

```shell
pip install --no-build-isolation git+https://github.com/Dao-AILab/fast-hadamard-transform.git
pip install git+https://github.com/tile-ai/tilelang
```

First convert huggingface model weights to the the format required by our inference demo. Set `MP` to match your available GPU count:
```bash
cd inference
export EXPERTS=256
export MP=8  # 根据实际 GPU 数调整 (如 4/2/1，需满足 256 % MP == 0)
export HF_CKPT_PATH=/data/models/deepseek-v3.2-exp
export SAVE_PATH=/data/models/deepseek-v3.2-exp-s
python convert.py --hf-ckpt-path ${HF_CKPT_PATH} --save-path ${SAVE_PATH} --n-experts ${EXPERTS} --model-parallel ${MP}
```

Launch the interactive chat interface and start exploring DeepSeek's capabilities:
```bash
export CONFIG=config_671B_v3.2.json
torchrun --nproc-per-node ${MP} generate.py --ckpt-path ${SAVE_PATH} --config ${CONFIG} --interactive
```