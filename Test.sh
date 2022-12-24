#!/usr/bin/env bash
for py_file in $(find Test -name *.py);
do
  python $py_file
done