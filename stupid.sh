#!/bin/bash

datadir="../data/asdf"
mkdir -p $datadir
echo $datadir
echo bye > "${datadir}/results.txt"
python stupid.py -m $datadir

# doesn't work
for i in {1..5}
do
	answer=$(bc <<< "scale=2;$i/5")
done