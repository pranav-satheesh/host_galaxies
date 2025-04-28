#!/bin/bash/

## for i in ${1..3..1}
##  do 
##    echo $i
##    sbatch submit_job.sh $i 
##  done

##for i in $(seq 9 1 25)
for i in $(seq 0 1 19)
do
   echo "Welcome $i times"
   sbatch submit_job.sh $i
done

#sbatch submit_job.sh 
#sbatch submit_job.sh 15
#sbatch submit_job.sh 7
