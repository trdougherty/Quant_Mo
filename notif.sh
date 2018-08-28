#!/usr/bin/env bash

while true;
do
NOTIF=`ps | grep python`
if [[ -z $NOTIF ]]; then
curl -X POST https://maker.ifttt.com/trigger/done_processing/with/key/hQLLOfkr58fFtcfG7NzkXJPLZjEM4bYytjjTYVPxQDw
break
fi
sleep 5
done
