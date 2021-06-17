#!/bin/bash
CUDA=1
MODEL='lenet'
MODEL1='mlp'

## baselines
# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40  --corrupted-test --model $MODEL --checkpoint --optimizer adam --method ensemble --n 10
# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40  --corrupted-test --model $MODEL1 --checkpoint --optimizer adam --method ensemble --n 10
CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40 --corrupted-test --model $MODEL --checkpoint --optimizer adam --method mcdrop --n 3 --dropout 0.1
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
# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40  --corrupted-test --model $MODEL --checkpoint --optimizer adam --method moe --n 2



