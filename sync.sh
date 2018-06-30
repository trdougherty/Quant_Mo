mkdir -p videos
sudo rsync -azP pi@raspberrypi.local:~/Quant_Mo/videos/$1/ ./videos/$1 --delete

