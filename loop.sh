##SET HOURS OF NORMAL OPERATION

START=7
END=19

while :
do
    current=$(date +%H)
    if [ $current -gt $START ] && [ $current -lt $END ]; then
    bash whole_process.sh $1
    sleep 10
    fi
done
