for i in `ls balance_result`
do
    if (( $(echo "$i > 0.04" |bc -l) ))
    then
        continue
    fi
    ./analysis.sh balance_result/$i | tee out
    echo $i:[`grep "Latency is" out|awk '{print $3 ","}'`], | tee -a data
    rm out
done
