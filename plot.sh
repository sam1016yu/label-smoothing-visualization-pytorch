#!/bin/bash



for i in {1..80}
do
python TSEN.py --ce --epoch $i
python TSEN.py --sim --epoch $i
python TSEN.py --epoch $i
done

for i in {1..80}
do
python plot_confusion.py --ce --epoch $i
python plot_confusion.py --sim --epoch $i
python plot_confusion.py --epoch $i
done