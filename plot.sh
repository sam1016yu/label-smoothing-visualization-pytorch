#!/bin/bash



for i in {1..80}
do
python TSEN.py --ce --epoch $i
python TSEN.py --sim --epoch $i
python TSEN.py --epoch $i
done