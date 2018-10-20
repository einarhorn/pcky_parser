#!/bin/sh
# Set 3rd argument to pa for parent_annotated pcfg
# Set 3rd argument to pa+oov for parent_annotated and unknown handled pcfg
python3 hw4_topcfg.py $1 $2 pa+oov
