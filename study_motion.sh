python motion_correlation.py 0 $1 > light.txt
python motion_correlation.py 1 $1 > motion.txt

hist -b 150 -x -f light.txt
hist -b 150 -x -f motion.txt
scatter -x motion.txt -y light.txt -s 70
