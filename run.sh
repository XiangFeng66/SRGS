#!/bin/bash
datasets=(
	"./output/bicycle"
	"./output/bonsai"
	"./output/counter"
	"./output/flowers"
	"./output/garden"
	"./output/kitchen"
	"./output/stump"
	"./output/treehill"
)
for dataset in "${datasets[@]}"; do
	python metrics.py -m "$dataset" 
done
