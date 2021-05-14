#!/bin/bash
CUDA=1
MODEL='lenet'

CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 20 --method ncensemble --n 3 --corrupted-test --model $MODEL --checkpoint
CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 20 --method ensemble --n 3 --corrupted-test --model $MODEL --checkpoint
CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 20 --method mcdrop --n 3 --corrupted-test --model $MODEL --checkpoint
CUDA_VISIBLE_DEVICES=$CUDA python main.py --lr 1e-4 --batch-size 128 --epochs 20 --method single --corrupted-test --model $MODEL --checkpoint

# python main.py --lr 1e-4 --cpu --batch-size 128 --epochs 20 --method ncensemble --n 3 --corrupted-test
# python main.py --lr 1e-4 --cpu --batch-size 128 --epochs 20 --method ensemble --n 3 --corrupted-test
# python main.py --lr 1e-4 --cpu --batch-size 128 --epochs 20 --method mcdrop --n 5 --corrupted-test
# python main.py --lr 1e-4 --cpu --batch-size 128 --epochs 20 --method single --corrupted-test