python motion_correlation.py 0 $1 > light.txt
python motion_correlation.py 1 $1 > motion.txt

echo ""
echo "LIGHT DATA"
echo ""
hist -b 60 -x -f light.txt
echo ""
echo "MOTION DATA"
echo ""
hist -b 60 -x -f motion.txt
scatter -x motion.txt -y light.txt -s 70
