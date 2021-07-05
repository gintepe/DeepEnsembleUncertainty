#!/bin/bash
CUDA=0
MODEL='lenet'
MODEL1='mlp'

## baselines
# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40  --corrupted-test --model $MODEL --checkpoint --optimizer adam --method ensemble --n 5
# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40  --corrupted-test --model $MODEL1 --checkpoint --optimizer adam --method ensemble --n 10
# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40 --corrupted-test --model $MODEL --checkpoint --optimizer adam --method mcdrop --n 5 --dropout 0.1
# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40 --corrupted-test --model $MODEL --checkpoint --optimizer adam --method single 

## ncens
# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40 --corrupted-test --model $MODEL --checkpoint --optimizer adam --method ncensemble --n 10 --reg-weight 5.0 --reg-decay 1.0
# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40 --corrupted-test --model $MODEL1 --checkpoint --optimizer adam --method ncensemble --n 10 --reg-weight 5.0 --reg-decay 1.0
# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40 --corrupted-test --model $MODEL --checkpoint --optimizer adam --method ncensemble --n 3 --reg-weight 10.0 --reg-decay 1.0
# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40 --corrupted-test --model $MODEL --checkpoint --optimizer adam --method ncensemble --n 3 --reg-weight 100.0 --reg-decay 0.8
# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40 --corrupted-test --model $MODEL --checkpoint --optimizer adam --method ncensemble --n 3 --reg-weight 10.0 --reg-decay 0.95

## ceens
# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40 --corrupted-test --model $MODEL --checkpoint --method ceensemble --n 10 --reg-weight 0.75 --reg-decay 1.0
# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40 --corrupted-test --model $MODEL1 --checkpoint --method ceensemble --n 10 --reg-weight 0.75 --reg-decay 1.0
# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40 --corrupted-test --model $MODEL --checkpoint --optimizer adam --method ceensemble --n 3 --reg-weight 1.0 --reg-decay 1.0
# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40 --corrupted-test --model $MODEL --checkpoint --optimizer adam --method ceensemble --n 3 --reg-weight 0.75 --reg-decay 1.0
# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40 --corrupted-test --model $MODEL --checkpoint --optimizer adam --method ceensemble --n 3 --reg-weight 0.5 --reg-decay 1.0

## MoE
# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40  --corrupted-test --model $MODEL --checkpoint --optimizer adam --method moe --moe-type sparse --predict-gated --n 2
# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40  --corrupted-test --model $MODEL --checkpoint --optimizer adam --method moe --moe-type fixed --n 5
# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40  --corrupted-test --model $MODEL --checkpoint --optimizer adam --method moe --moe-type fixed --n 10

# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40  --corrupted-test --model $MODEL --checkpoint --optimizer adam --method moe --moe-type sparse --predict-gated --n 2
# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40  --corrupted-test --model $MODEL --checkpoint --optimizer adam --method moe --moe-type sparse --predict-gated --n 5
# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40  --corrupted-test --model $MODEL --checkpoint --optimizer adam --method moe --moe-type sparse --predict-gated --n 10

# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40  --corrupted-test --model $MODEL --checkpoint --optimizer adam --method moe --moe-type sparse --predict-gated --moe-topk 2 --n 2
# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40  --corrupted-test --model $MODEL --checkpoint --optimizer adam --method moe --moe-type sparse --predict-gated --moe-topk 2 --n 5
# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40  --corrupted-test --model $MODEL --checkpoint --optimizer adam --method moe --moe-type sparse --predict-gated --moe-topk 2 --n 10

# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40  --corrupted-test --model $MODEL --checkpoint --optimizer adam --method moe --moe-type sparse --n 2
# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40  --corrupted-test --model $MODEL --checkpoint --optimizer adam --method moe --moe-type sparse --n 5
# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40  --corrupted-test --model $MODEL --checkpoint --optimizer adam --method moe --moe-type sparse --n 10

# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40  --corrupted-test --model $MODEL --checkpoint --optimizer adam --method moe --moe-type sparse --moe-topk 2 --n 2
# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40  --corrupted-test --model $MODEL --checkpoint --optimizer adam --method moe --moe-type sparse --moe-topk 2 --n 5
# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40  --corrupted-test --model $MODEL --checkpoint --optimizer adam --method moe --moe-type sparse --moe-topk 2 --n 10

# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40  --corrupted-test --model $MODEL --dropout 0.9 --reg-weight 0.1 --checkpoint --optimizer adam --method moe --moe-type dense --moe-gating mcdc_simple --n 5 --predict-gated
# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40  --corrupted-test --model $MODEL --dropout 0.9 --reg-weight 0.1 --checkpoint --optimizer adam --method moe --moe-type dense --moe-gating mcd_simple --n 5

CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40  --corrupted-test --model $MODEL --dropout 0.5 --reg-weight 0.01 --checkpoint --optimizer adam --method moe --moe-type dense --moe-gating mcd_lenet --n 5 --moe-topk 2
CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40  --corrupted-test --model $MODEL --dropout 0.5 --reg-weight 0.01 --checkpoint --optimizer adam --method moe --moe-type dense --moe-gating mcd_lenet --n 5 --moe-topk 2 --predict-gated
CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40  --corrupted-test --model $MODEL --dropout 0.2 --reg-weight 0.01 --checkpoint --optimizer adam --method moe --moe-type dense --moe-gating mcd_lenet --n 5 --moe-topk 2
CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40  --corrupted-test --model $MODEL --dropout 0.2 --reg-weight 0.01 --checkpoint --optimizer adam --method moe --moe-type dense --moe-gating mcd_lenet --n 5 --moe-topk 2 --predict-gated

# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40  --corrupted-test --model $MODEL --checkpoint --optimizer adam --method moe --predict-gated --moe-type dense --moe-gating same --n 2
# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40  --corrupted-test --model $MODEL --checkpoint --optimizer adam --method moe --moe-type dense --moe-gating same
# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40  --corrupted-test --model $MODEL --dropout 0.9 --checkpoint --optimizer adam --method moe --moe-type dense --moe-gating simple --n 5
# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40  --corrupted-test --model $MODEL --checkpoint --optimizer adam --method moe --predict-gated --moe-topk 2 --moe-type dense --moe-gating simple --n 5

# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40  --corrupted-test --model $MODEL --checkpoint --optimizer adam --method moe --moe-type sparse --moe-gating simple --n 5
# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40  --corrupted-test --model $MODEL --checkpoint --optimizer adam --method moe --moe-type dense --moe-gating simple --n 5
# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40  --corrupted-test --model $MODEL --checkpoint --optimizer adam --method moe --moe-topk 2 --moe-type sparse --moe-gating simple --n 5
# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40  --corrupted-test --model $MODEL --checkpoint --optimizer adam --method moe --moe-topk 2 --moe-type dense --moe-gating simple --n 5
