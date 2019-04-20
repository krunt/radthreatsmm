#!/bin/bash
set -x

for i in {0..1} ; do
    mypython3 main_multi.py $i 2 &
done

wait

head -1 ../submission0.csv > ../submission.csv
for i in {0..1} ; do
    sed '1d' ../submission$i.csv >> ../submission.csv
done
    
