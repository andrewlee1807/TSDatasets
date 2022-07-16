#!/bin/sh
python test.py > export.out 2>&1

#Run background with specified name
bash -c "exec -a Andrew python fedot_test.py &"