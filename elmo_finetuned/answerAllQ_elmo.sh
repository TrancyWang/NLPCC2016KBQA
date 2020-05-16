#!/bin/bash

. ~/.bashrc
cd /share/data/lang/users/zeweichu/projs/kbqa/NLPCC2016KBQA/elmo_finetuned
conda activate transformers

python core_elmo.py ../nlpcc-iccpol-2016.kbqa.testing-data answer_elmo_cosine ../full_dataset/kbJson.cleanPre.alias.utf8 ../full_dataset/outputAP ../full_dataset/countChar ../full_dataset/vectorJson.utf8 1 30
