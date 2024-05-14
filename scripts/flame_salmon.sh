export CUDA_VISIBLE_DEVICES=$1
python3 train.py --config configs/dynerf/flame_salmon_sub0.yaml
python3 train.py --config configs/dynerf/flame_salmon_sub1.yaml
python3 train.py --config configs/dynerf/flame_salmon_sub2.yaml
python3 train.py --config configs/dynerf/flame_salmon_sub3.yaml