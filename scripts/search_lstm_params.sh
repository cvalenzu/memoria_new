#!/bin/bash

basePath=../data/
paths=( canela.csv  monte_redondo.csv  totoral.csv )

inputs=( 1 6 12 24 )
output=12
timesteps=( 12 24 48 72 168 336 )
batch_size=( 50 100 150 200 500 1000 )
epoch=10
preprocess=( minmax_1 minmax_2 std )
stateful=( true false )

for path in ${paths[@]};do
	for input in ${inputs[@]};do 
		for timestep in ${timesteps[@]};do
			for batch_size in ${batch_size[@]};do
				for preproc in ${preprocess[@]};do
					for state in ${stateful[@]};do
						python ../code/lstm_selecting_params.py $basePath$path --inputs $input --outputs $output --timesteps $timestep --batch_size $batch_size --epochs $epoch --preprocess $preproc --stateful $state	
					done
				done
			done
		done
	done 
done
