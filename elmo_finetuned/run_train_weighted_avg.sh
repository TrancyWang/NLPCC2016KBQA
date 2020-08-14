#!/bin/bash

. ~/.bashrc
cd /share/data/lang/users/zeweichu/projs/kbqa/NLPCC2016KBQA/elmo_finetuned
conda activate transformers

python main_weighted_avg.py \
    --loss_function $1
