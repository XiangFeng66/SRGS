#!/bin/bash
datasets=(
	"./output/bonsai"
	"./output/counter"
	"./output/flowers"
	"./output/garden"
	"./output/kitchen"
	"./output/stump"
	"./output/treehill"
)
for dataset in "${datasets[@]}"; do
	python render.py -m "$dataset" -r 8 --skip_train
done
