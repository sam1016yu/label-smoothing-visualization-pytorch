#!/bin/bash



for i in {31..120}
do
python plot_confusion.py --ce --epoch $i
python plot_confusion.py --sim --epoch $i
python plot_confusion.py --epoch $i
done