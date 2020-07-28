for s in 123 234 345 456 567 678 789 890 901 12	23 34 45 56 67 78 89 90 1 100
do
    CUDA_VISIBLE_DEVICES=2 python3 -u main.py 40 .1 random $s > outfiles/final/lr.1_random_$s.txt
    CUDA_VISIBLE_DEVICES=2 python3 -u main.py 40 .01 random $s > outfiles/final/lr.01_random_$s.txt
    CUDA_VISIBLE_DEVICES=2 python3 -u main.py 40 .001 random $s > outfiles/final/lr.001_random_$s.txt
    CUDA_VISIBLE_DEVICES=2 python3 -u main.py 40 .0001 random $s > outfiles/final/lr.0001_random_$s.txt
    CUDA_VISIBLE_DEVICES=2 python3 -u main.py 40 .00001 random $s > outfiles/final/lr.00001_random_$s.txt
done