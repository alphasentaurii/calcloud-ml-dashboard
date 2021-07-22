#!/bin/bash -xu

DATES=('2021-05-11-1620740441' '2021-05-13-1620929899' '2021-05-15-1621096666')
BUCKET=calcloud-modeling-sb
RESCLF=('duration' 'history' 'kfold' 'matrix' 'preds' 'proba' 'scores' 'y_pred' 'y_true')
RESREG=('duration' 'history' 'kfold' 'predictions' 'residuals' 'scores')


for d in "${DATES[@]}"
do
	clfpath=`echo ${d}/results/mem_bin`
	mkdir -p $clfpath
	for r in "${RESCLF[@]}"

	do
		aws s3api get-object --bucket $BUCKET --key ${clfpath}/${r} ${clfpath}/${r}
	done

	mempath=`echo ${d}/results/memory`
	wallpath=`echo ${d}/results/wallclock`
	mkdir -p $mempath && mkdir -p $wallpath

	for r in "${RESREG[@]}"

	do
		aws s3api get-object --bucket $BUCKET --key ${mempath}/${r} ${mempath}/${r}
		aws s3api get-object --bucket $BUCKET --key ${wallpath}/${r} ${wallpath}/${r}
	done
	
	model_path=`echo ${d}`
	aws s3api get-object --bucket $BUCKET --key ${d}/models/models.zip ${model_path}/models.zip
done
