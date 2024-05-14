export CUDA_VISIBLE_DEVICES=$1
ulimit -Sn 100000
python3 train.py --config configs/mobile_stage/dance3_sub0.yaml
python3 train.py --config configs/mobile_stage/dance3_sub1.yaml
python3 train.py --config configs/mobile_stage/dance3_sub2.yaml
python3 train.py --config configs/mobile_stage/dance3_sub3.yaml