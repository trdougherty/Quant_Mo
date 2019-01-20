python motion_correlation.py 0 $1 > light.txt
python motion_correlation.py 1 $1 > motion.txt

hist -x -f light.txt
hist -x -f motion.txt
