
while :
do
sudo rsync -azP pi@raspberrypi.local:~/Quant_Motion_Analysis/tests/ ./test_files
sleep 30
done
