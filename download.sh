#!/bin/bash

download_from_gdrive() {
	file_id=$1
	file_name=$2

	# first stage to get the warning html
	curl -c /tmp/cookies \
	"https://drive.google.com/uc?export=download&id=$file_id" > \
	/tmp/intermezzo.html

	# second stage to extract the download link from html above
	download_link=$(cat /tmp/intermezzo.html | \
	grep -Po 'uc-download-link" [^>]* href="\K[^"]*' | \
	sed 's/\&amp;/\&/g')
	curl -L -b /tmp/cookies \
	"https://drive.google.com$download_link" > $file_name
}

if [ ! -f ./v2/ckpt/blstm/epoch_90.pt ]; then
	download_from_gdrive 12e6XaEhmnAJRvxn8TMxdKZPaSVOoABBv ./v2/ckpt/blstm/epoch_90.pt
fi

download_from_gdrive 12e6XaEhmnAJRvxn8TMxdKZPaSVOoABBv ./v2/ckpt/blstm/epoch_90.pt


