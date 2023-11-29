#!/bin/bash
for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

# use here your expected variables
if [[ $SCENARIO=="universal" ]]
then 
    if [[ $ENVIRONMENT=="constant" ]]
    then
        for i in $(seq 1 $ITERS); 
        do python3 fine_tuned/vmas/vmas_run.py task=vmas/simple_reference_const algorithm=maddpg algorithm.share_param_critic=false; 
        done;
    else
        for i in $(seq 1 $ITERS); 
        do python3 fine_tuned/vmas/vmas_run.py task=vmas/simple_reference algorithm=maddpg algorithm.share_param_critic=false; 
        done;
    fi
elif [[ $SCENARIO=="noise" ]]
then 
    if [[ $ENVIRONMENT=="constant" ]]
    then
        for i in $(seq 1 $ITERS); 
        do python3 fine_tuned/vmas/vmas_run.py task=vmas/simple_reference_idiolect_const algorithm=maddpg algorithm.share_param_critic=false; 
        done;
    else
        for i in $(seq 1 $ITERS); 
        do python3 fine_tuned/vmas/vmas_run.py task=vmas/simple_reference_idiolect algorithm=maddpg algorithm.share_param_critic=false; 
        done;
    fi
elif [[ $SCENARIO=="memory" ]]
then 
    if [[ $ENVIRONMENT=="constant" ]]
    then
        for i in $(seq 1 $ITERS); 
        do python3 fine_tuned/vmas/vmas_run.py task=vmas/simple_reference_idiolect_mem_buffer_const algorithm=maddpg algorithm.share_param_critic=false; 
        done;
    else
        for i in $(seq 1 $ITERS); 
        do python3 fine_tuned/vmas/vmas_run.py task=vmas/simple_reference_idiolect_mem_buffer algorithm=maddpg algorithm.share_param_critic=false; 
        done;
    fi
elif [[ $SCENARIO=="both" ]]
then 
    if [[ $ENVIRONMENT=="constant" ]]
    then
        for i in $(seq 1 $ITERS); 
        do python3 fine_tuned/vmas/vmas_run.py task=vmas/simple_reference_idiolect_noise_mem_const algorithm=maddpg algorithm.share_param_critic=false; 
        done;
    else
        for i in $(seq 1 $ITERS); 
        do python3 fine_tuned/vmas/vmas_run.py task=vmas/simple_reference_idiolect_noise_mem algorithm=maddpg algorithm.share_param_critic=false; 
        done;
    fi
else
echo "Please select valid scenario"
fi