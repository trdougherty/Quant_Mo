#!/usr/bin/env bash

mkdir -p videos
rsync -avzu -e "ssh -i /Users/TRD/.ssh/id_rsa" --delete pi@raspberrypi.local:~/Quant_Mo/videos/$1/ ./videos/$1
