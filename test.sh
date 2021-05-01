#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python main.py --lr 1e-4 --batch-size 128 --epochs 20 --method ensemble --n 5 --corrupted-test
CUDA_VISIBLE_DEVICES=1 python main.py --lr 1e-4 --batch-size 128 --epochs 20 --method ensemble --n 3 --corrupted-test
CUDA_VISIBLE_DEVICES=1 python main.py --lr 1e-4 --batch-size 128 --epochs 20 --method mcdrop --n 5 --corrupted-test
CUDA_VISIBLE_DEVICES=1 python main.py --lr 1e-4 --batch-size 128 --epochs 20 --method single --corrupted-test