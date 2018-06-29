#! /bin/bash

### USAGE ###
# bash whole_process.sh name_of_exp

#ENDINGS FOR SAVING CONVENTION
avi=.avi
motion=_motion.avi
#compile=_compiled.jpg

GEN_NAME=$(cat name)
time_s=$(date +%Y-%m-%d.%H:%M:%S)

if [[ -n $1 ]];then
    name=$1
    echo "Name of process is $1"
    mkdir -p ./videos
    mkdir -p ./videos/$1
    storage=./videos/$1/
else
    echo "Please indicate name of this process"
    exit 1
fi

python record.py -t 10 -o $storage$name$avi
python video_application.py -i $storage$name$avi -o $storage$name$motion -r $storage$name'@'$time_s
rm $storage$name$avi

#python video_compression.py -i $storage$name$motion -o $storage$name$compile
