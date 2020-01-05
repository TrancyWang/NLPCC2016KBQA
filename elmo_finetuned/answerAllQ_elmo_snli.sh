#!/bin/bash
source activate py37
 python core_elmo_snli.py ../nlpcc-iccpol-2016.kbqa.testing-data answer_elmo_snli ../full_dataset/kbJson.cleanPre.alias.utf8 ../full_dataset/outputAP ../full_dataset/countChar ../full_dataset/vectorJson.utf8 1 30
# python core_elmo_snli.py ../nlpcc-iccpol-2016.kbqa.testing-data answer_elmo_snli ../kbJson.cleanPre.alias.utf8 ../outputAP ../countChar ../vectorJson.utf8 1 30
