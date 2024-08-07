#!/bin/bash

# Create logs directory if it doesn't exist
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

# Set sequence length and model name
seq_len=512
model_name=TSMixer

# Declare dictionaries for hyperparameters
declare -A learning_rate
declare -A num_blocks
declare -A dropout
declare -A hidden_size

# Best hyperparameters for each prediction length
learning_rate[96]=0.0001
num_blocks[96]=6
dropout[96]=0.9
hidden_size[96]=512

learning_rate[192]=0.001
num_blocks[192]=4
dropout[192]=0.9
hidden_size[192]=256

learning_rate[336]=0.001
num_blocks[336]=4
dropout[336]=0.9
hidden_size[336]=256

learning_rate[720]=0.001
num_blocks[720]=2
dropout[720]=0.9
hidden_size[720]=64

# Loop through prediction lengths and run experiments with best hyperparameters
for pred_len in 96 192 336 720
do
  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_${seq_len}_${pred_len} \
    --model $model_name \
    --data ETTh1 \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --num_blocks ${num_blocks[$pred_len]} \
    --dropout ${dropout[$pred_len]} \
    --hidden_size ${hidden_size[$pred_len]} \
    --learning_rate ${learning_rate[$pred_len]} \
    --activation relu \
    --individual False \
    --affine True \
    --loss mse \
    --des 'Exp with TSMixer ETTh1' \
    > ./logs/LongForecasting/ETTh1_${seq_len}_${pred_len}.log
done
