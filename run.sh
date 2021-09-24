#!/bin/bash

make build
rm -rf ./temp_out
mkdir temp_out

rm analysisFile.out

for i in 700000 1000000 
do 
    for t in 1 2 4 8 16 17 18 19 20
    do  
        for j in 1
        do
            ./goi-parallel.out ./sample_inputs_varies_gen/${i}.in ./temp_out/${i}_thread${t}.out $t
        done
    done
done