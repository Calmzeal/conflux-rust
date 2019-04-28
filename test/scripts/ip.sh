#sudo rm -rf /etc/hosts
region=us-west-1
mv ips ips_old
touch ips
if [[ -f instances ]]
then
	instance=`cat instances`
	response=`aws ec2 describe-instances --instance-ids $instance --region $region`
	echo $response | jq ".Reservations[].Instances[].PrivateIpAddress" | tr -d '"' > ips_tmp
	#echo $response | jq ".Reservations[].Instances[].PublicIpAddress" | tr -d '"' > ips_tmp
	uniq ips_tmp > ips
	rm ips_tmp
fi
echo GET `wc -l ips` IPs
exit
ips=(`cat ips`)
for i in `seq 0 $((${#ips[@]}-1))`
do
  echo ${ips[$i]} n$i |sudo tee -a /etc/hosts
  ssh -o "StrictHostKeyChecking no" n$i &
done
wait
#scp ips_current lpl@blk:~/ips
#scp -i MyKeyPair.pem ips_current ubuntu@aws:~/ssd/ips
#ssh vm "./tmp.sh"
#rm ipof*
