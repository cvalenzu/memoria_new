#!/bin/bash

FILES=( ../data/canela.csv ../data/totoral.csv ../data/monte_redondo.csv )
RESULTS=( ../results/best_params/canela_best_params_by_lag.csv ../results/best_params/totoral_best_params_by_lag.csv ../results/best_params/monte_best_params_by_lag.csv )

EPOCHS=100

cd ../code
for index in ${!FILES[*]};do
        file=${FILES[$index]}
        result=${RESULTS[$index]}
        echo "Processing $file"
        python train_best_lstm_by_lag.py $file $result --epochs $EPOCHS 
done

