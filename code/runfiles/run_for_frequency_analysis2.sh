for s in 123 234 345 456 567 678 789 890 901 12 23 34 45 56 67 78 89 90 1 100
do
    CUDA_VISIBLE_DEVICES=1 python3 -u main.py 40 .001 plasticity $s > outfiles/plasticity/lr.001_plasticity_$s.txt
    CUDA_VISIBLE_DEVICES=1 python3 -u main.py 40 .001 sn $s > outfiles/subjective_novelty/lr.001_SN_$s.txt
done