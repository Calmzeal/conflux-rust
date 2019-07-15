idx=0
for i in `ls $1/*`
do
    if [[ $i > 10 ]]
    then
        echo $i
        log_dir=(`grep Initializing $i|awk '{print $7}'`)
        final_block=(`grep Merged $i|awk '{print $10}'`)
        aim_block=(`grep Merged $i|awk '{print $6}'`)
        echo ${log_dir[$idx]} ${final_block[$idx]} ${aim_block[$idx]}
        ./stat_latency.py ${log_dir[$idx]} ${final_block[$idx]} ${aim_block[$idx]}
    fi
done
