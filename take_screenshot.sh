#!/bin/bash
export DISPLAY=:0
curl -s -X PUT http://localhost:9222/json/new
UDD=$(cat /tmp/udd.txt)
WID=$(xwininfo -root -tree | grep "$UDD" | grep -v xwininfo | awk '{print $1}' | head -n 1)
SS=$(mktemp --suffix=.png)
echo "Screenshot path: $SS"
import -window "$WID" "$SS"
