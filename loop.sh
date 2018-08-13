##SET HOURS OF NORMAL OPERATION

START=9
END=18

while :
do
    current=$(date +%H)
    if [ $current -gt $START ] && [ $current -lt $END ]; then
    do
    bash whole_process.sh $1
    sleep 10
    fi
done
