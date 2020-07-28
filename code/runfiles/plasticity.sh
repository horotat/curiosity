for s in 123 234 345 456 567 678 789 890 901 12 23 34 45 56 67 78 89 90 1 100
do
    CUDA_VISIBLE_DEVICES=1 python3 -u main.py 40 .1 curious $s plasticity > outfiles/plasticity/lr.1_curious_plasticity_$s.txt
    CUDA_VISIBLE_DEVICES=1 python3 -u main.py 40 .01 curious $s plasticity > outfiles/plasticity/lr.01_curious_plasticity_$s.txt
    CUDA_VISIBLE_DEVICES=1 python3 -u main.py 40 .001 curious $s plasticity > outfiles/plasticity/lr.001_curious_plasticity_$s.txt
    CUDA_VISIBLE_DEVICES=1 python3 -u main.py 40 .0001 curious $s plasticity > outfiles/plasticity/lr.0001_curious_plasticity_$s.txt
    CUDA_VISIBLE_DEVICES=1 python3 -u main.py 40 .00001 curious $s plasticity > outfiles/plasticity/lr.00001_curious_plasticity_$s.txt
done
