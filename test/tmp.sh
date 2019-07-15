for i in 02 04 06 08
do
    echo Start $i
    rate=0.$i
    mkdir balance_result/$rate
    for j in `seq 3`
    do
        ./balance_attack_basic.py --evil $rate | tee balance_result/$rate/$j &
    done
    wait
done
