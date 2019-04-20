#!/bin/bash
set -x

mypython3 hist_prepare_multi_precision.py 4 8 &
mypython3 hist_prepare_multi_precision.py 5 8 &
wait

mypython3 hist_prepare_multi_precision.py 6 8 &
mypython3 hist_prepare_multi_precision.py 7 8 &
wait

