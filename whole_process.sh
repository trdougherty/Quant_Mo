#! /bin/bash

#ENDINGS FOR SAVING CONVENTION
avi=.avi
motion=_motion.avi
compile=_compiled.jpg

if [[ -n $1 ]];then
    name=$1
    echo "Name of process is $1"
    if [[ -d videos ]];then
        mkdir ./videos/$1
        storage=./videos/$1/
    else
        echo "Please verify video storage location"
        exit 2
    fi
else
    echo "Please indicate name of this process"
    exit 1
fi

python record.py -t 10 -o $storage$name$avi
python video_application.py -i $storage$name$avi -o $storage$name$motion
python video_compression.py -i $storage$name$motion -o $storage$name$compile
