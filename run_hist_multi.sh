#!/bin/bash
set -x

for i in {0..7} ; do
    mypython3 hist_prepare_multi.py $i 8 &
done

wait
