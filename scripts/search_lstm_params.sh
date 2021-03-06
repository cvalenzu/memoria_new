#!/bin/bash

basePath=../data/
paths=( canela.csv  monte_redondo.csv  totoral.csv )

inputs=( 1 24 12 6 )
output=12
timesteps=( 1 24 72 168 )
batch_size=( 1200 )
epoch=10
preprocess=( minmax_1 minmax_2 std )
stateful=( false true )


for timestep in ${timesteps[@]};do
	for input in ${inputs[@]};do
		for path in ${paths[@]};do
			for batch_size in ${batch_size[@]};do
				for preproc in ${preprocess[@]};do
					for state in ${stateful[@]};do
						python ../code/lstm_selecting_params.py $basePath$path --verbose true --inputs $input --outputs $output --timesteps $timestep --batch_size $batch_size --epochs $epoch --preprocess $preproc --stateful $state	
					done
				done
			done
		done
	done 
done
