ulimit -Sn 100000
CUDA_VISIBLE_DEVICES=$1 python3 train.py --config configs/enerf_outdoor/actor5_6_sub0_2.yaml >sub0_2.out 2>&1 & 
CUDA_VISIBLE_DEVICES=$2 python3 train.py --config configs/enerf_outdoor/actor5_6_sub1_2.yaml >sub1_2.out 2>&1 & 
CUDA_VISIBLE_DEVICES=$3 python3 train.py --config configs/enerf_outdoor/actor5_6_sub2_2.yaml >sub2_2.out 2>&1 & 
CUDA_VISIBLE_DEVICES=$4 python3 train.py --config configs/enerf_outdoor/actor5_6_sub3_2.yaml >sub3_2.out 2>&1 &