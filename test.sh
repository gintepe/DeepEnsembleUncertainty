#!/bin/bash
CUDA=1
MODEL='lenet'
MODEL1='mlp'
# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 20  --corrupted-test --model $MODEL --checkpoint --optimizer adam --method ncensemble --n 3 
CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40  --corrupted-test --model $MODEL --checkpoint --optimizer adam --method ensemble --n 3
CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40  --corrupted-test --model $MODEL1 --checkpoint --optimizer adam --method ensemble --n 3
# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40  --corrupted-test --model $MODEL --checkpoint --optimizer adam --method mcdrop --n 3
# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40  --corrupted-test --model $MODEL --checkpoint --optimizer adam --method single 
# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40 --corrupted-test --model $MODEL --checkpoint --method ceensemble --n 3 --reg-weight 0.75 --reg-decay 1.0
# python main.py --lr 1e-4 --cpu --batch-size 128 --epochs 20 --method ncensemble --n 3 --corrupted-test
# python main.py --lr 1e-4 --batch-size 128 --epochs 40 --model $MODEL --method ensemble --n 3 --corrupted-test
# python main.py --lr 1e-4 --batch-size 128 --epochs 40 --model $MODEL --method mcdrop --n 3 --corrupted-test
# python main.py --lr 1e-4 --batch-size 128 --epochs 40 --model $MODEL --method single --corrupted-test

# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40 --corrupted-test --model $MODEL --checkpoint --method ncensemble --n 3 --reg-weight 2.0 --reg-decay 0.95
# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40 --corrupted-test --model $MODEL --checkpoint --method ncensemble --n 3 --reg-weight 10.0 --reg-decay 0.9
# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 20 --method ncensemble --n 3 --corrupted-test --model $MODEL --checkpoint --reg-weight 100.0

# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 20 --method ncensemble --n 3 --corrupted-test --model $MODEL1 --checkpoint --reg-weight 0.1
# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 20 --method ncensemble --n 3 --corrupted-test --model $MODEL1 --checkpoint --reg-weight 0.3
# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 20 --method ncensemble --n 3 --corrupted-test --model $MODEL1 --checkpoint --reg-weight 0.7
# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 20 --method ncensemble --n 3 --corrupted-test --model $MODEL1 --checkpoint --reg-weight 1.0
# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 20 --method ncensemble --n 3 --corrupted-test --model $MODEL1 --checkpoint --reg-weight 2.0

# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40 --corrupted-test --model $MODEL --checkpoint --optimizer adam --method ncensemble --n 3 --reg-weight 5.0 --reg-decay 1.0
# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40 --corrupted-test --model $MODEL --checkpoint --optimizer adam --method ncensemble --n 3 --reg-weight 10.0 --reg-decay 1.0
# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40 --corrupted-test --model $MODEL --checkpoint --optimizer adam --method ncensemble --n 3 --reg-weight 100.0 --reg-decay 0.8
# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40 --corrupted-test --model $MODEL --checkpoint --optimizer adam --method ncensemble --n 3 --reg-weight 10.0 --reg-decay 0.95


# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40 --corrupted-test --model $MODEL1 --checkpoint --optimizer adam --method ceensemble --n 3 --reg-weight 1.0 --reg-decay 1.0
# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40 --corrupted-test --model $MODEL1 --checkpoint --optimizer adam --method ceensemble --n 3 --reg-weight 0.75 --reg-decay 1.0
# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40 --corrupted-test --model $MODEL1 --checkpoint --optimizer adam --method ceensemble --n 3 --reg-weight 0.5 --reg-decay 1.0
# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40 --corrupted-test --model $MODEL1 --checkpoint --optimizer adam --method ceensemble --n 3 --reg-weight 0.25 --reg-decay 1.0
# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40 --corrupted-test --model $MODEL1 --checkpoint --optimizer adam --method ceensemble --n 3 --reg-weight 10.0 --reg-decay 0.95

# CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 40 --corrupted-test --model $MODEL --checkpoint --optimizer adam --method ceensemble --n 3 --reg-weight 10.0 --reg-decay 0.95



