#!/bin/bash

num_jobs=$1

for (( i = 1; i <= num_jobs; i++ ))
do
    CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $(( 12345 + i )) model-deepspeed.py \
    --deepspeed --deepspeed_config ds_config.json &> "job${i}.out" &
done

