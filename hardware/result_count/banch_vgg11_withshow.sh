#!/bin/bash

#---------------------------------------------------------------------
# Script variables
#---------------------------------------------------------------------
#file_num_start=$1
#file_num_end=$2
#pack_size=$3
file_num_start=0
file_num_end=419
pack_size=500
#if you want to send once, just make pack_size greater than file_num_end - file_num_start
tool_path=../tools

# Size of PCIe DMA transfers that will be used for this test.
# Make sure valid addresses exist in the FPGA when modifying this
# variable. Addresses in the range of 0 - (4 * transferSize) will  
# be used for this test when the PCIe DMA core is setup for memory
# mapped transaction.

# Set the number of times each data transfer will be repeated.
# Increasing this number will allow transfers to accross multiple
# channels to over lap for a longer period of time.
transferCount=1

# Determine which Channels are enabled
# Determine if the core is Memory Mapped or Streaming
isStreaming=0
h2cChannels=0
for ((i=0; i<=3; i++)); do
	v=`$tool_path/reg_rw /dev/xdma0_control 0x0${i}00 w`
	returnVal=$?
	if [ $returnVal -ne 0 ]; then
		break;
	fi

	#v=`echo $v | grep -o  '): 0x[0-9a-f]*'`
	statusRegVal=`$tool_path/reg_rw /dev/xdma0_control 0x0${i}00 w | grep "Read.*:" | sed 's/Read.*: 0x\([a-z0-9]*\)/\1/'`
	channelId=${statusRegVal:0:3}
	streamEnable=${statusRegVal:4:1}

	if [ $channelId == "1fc" ]; then
		h2cChannels=$((h2cChannels + 1))
		if [ $streamEnable == "8" ]; then
			isStreaming=1
		fi
	fi
done
echo "Info: Number of enabled h2c channels = $h2cChannels"

# Find enabled c2hChannels
c2hChannels=0
for ((i=0; i<=3; i++)); do
	v=`$tool_path/reg_rw /dev/xdma0_control 0x1${i}00 w`
	returnVal=$?
	if [ $returnVal -ne 0 ]; then
		break;
	fi

	$tool_path/reg_rw /dev/xdma0_control 0x1${i}00 w | grep "Read.*: 0x1fc" > /dev/null
	statusRegVal=`$tool_path/reg_rw /dev/xdma0_control 0x1${i}00 w | grep "Read.*:" | sed 's/Read.*: 0x\([a-z0-9]*\)/\1/'`
	channelId=${statusRegVal:0:3}

	# there will NOT be a mix of MM & ST channels, so no need to check
	# for streaming enabled
	if [ $channelId == "1fc" ]; then
		c2hChannels=$((c2hChannels + 1))
	fi
done
echo "Info: Number of enabled c2h channels = $c2hChannels"

# Report if the PCIe DMA core is memory mapped or streaming
if [ $isStreaming -eq 0 ]; then
	echo "Info: The PCIe DMA core is memory mapped."
else
	echo "Info: The PCIe DMA core is streaming."
fi

# Check to make sure atleast one channel was identified
if [ $h2cChannels -eq 0 -a $c2hChannels -eq 0 ]; then
	echo "Error: No PCIe DMA channels were identified."
	exit 1
fi

channelPairs=$(($h2cChannels < $c2hChannels ? $h2cChannels : $c2hChannels))
if [ $channelPairs -gt 0 ]; then
    # ./dma_streaming_test.sh $transferSize $transferCount $channelPairs
    echo "Info: Running PCIe DMA streaming test"
    echo "      transfer size:  $transferSize_send"
    echo "      transfer count: $transferCount"
    echo "Info: Only channels that have both h2c and c2h will be tested as the other"
    echo "      interfaces are left unconnected in the PCIe DMA example design. "
	totle_num=$[$file_num_end-$file_num_start]
	if [ $totle_num -gt $pack_size ]; then
		send_time=$[($file_num_end-$file_num_start)/$pack_size]
                tmp=$[$file_num_end-$file_num_start-($send_time*$pack_size)]
		#if [ $tmp -eq 0 ]; then 
                #    send_time=$[$send_time - 1] 
                #fi
		#echo "file numer greater than $pack_size, send $send_time times"
		send_start=0
		send_end=$pack_size
		rm -f data/output_bin/*
		for ((i=0; i<$send_time; i++));
		do
			transferSize_send=$[($send_end-$send_start+1)*224*224*3*8/8+64/8]
			transferSize_rcv=$[($send_end-$send_start+1)*64/8]
			rm -f ./data/input_bin/tmp/*
                        # bu zeros
                        send_num=$[$send_end-$send_start]
                        file_pose=`echo "obase=16; $send_num"|bc`
                        len=${#file_pose}
                        pose_zero=$[16-$len]
                        for((k=0;k<$pose_zero;k++)); 
                        do
                            file_pose='0'$file_pose
                        done
                        file_pose_revr=''
                        for((i=0;i<8;i++)); do
                            strtmp=${file_pose:$[(7-$i)*2]:2}
                            file_pose_revr=$file_pose_revr$strtmp
                            done
                        echo $file_pose_revr
                        echo $file_pose_revr | xxd -r -ps >> ./data/input_bin/tmp/tmp${i}.bin
			for((j=$send_start; j<$send_end; j++));
			do
				cat ./data/input_bin/img${j}.bin >> ./data/input_bin/tmp/tmp${i}.bin
			done
			$tool_path/dma_from_device -d /dev/xdma0_c2h_0 -f ./data/output_bin/vggoutput_pack${i}.bin -s $transferSize_rcv -c $transferCount &
			sleep 1
			$tool_path/dma_to_device -d /dev/xdma0_h2c_0 -f ./data/input_bin/tmp/tmp${i}.bin -s $transferSize_send -c $transferCount &
			sleep 1
			send_start=$send_end
			send_end=$((i<$[send_time-1]?$[$send_end+$pack_size]:$file_num_end))
			echo "$send_start, $send_end"
		done
		for ((i=0; i<$send_time; i++));
		do
			cat ./data/output_bin/vggoutput_pack${i}.bin >> ./data/output_bin/vggoutput_pack.bin
		done
		#echo "trans times end"
	else
		transferSize_send=$[($file_num_end-$file_num_start+1)*224*224*3*8/8+64/8]
		transferSize_rcv=$[($file_num_end-$file_num_start+1)*64/8]
		#echo "file numer less than $pack_size, send once"
		rm -f ./data/input_bin/tmp/tmp.bin
                # bu zeros
                send_num=$[$file_num_end-$file_num_start+1]
                file_pose=`echo "obase=16; $send_num"|bc`
                len=${#file_pose}
                pose_zero=$[16-$len]
                for((k=0;k<$pose_zero;k++)); 
                do
                    file_pose='0'$file_pose
                done
                file_pose_revr=''
                for((i=0;i<8;i++)); do
                    strtmp=${file_pose:$[(7-$i)*2]:2}
                    file_pose_revr=$file_pose_revr$strtmp
                done
                echo $file_pose_revr
                echo $file_pose_revr | xxd -r -ps >> ./data/input_bin/tmp/tmp.bin
		for ((i=$file_num_start; i<=$file_num_end; i++)); 
		do
			cat ./data/input_bin/img${i}.bin >> ./data/input_bin/tmp/tmp.bin
		done
		rm -f data/output_bin/vggoutput_pack.bin
		# echo "Info: DMA setup to read from c2h channel $i. Waiting on write data to channel $i."
		$tool_path/dma_from_device -d /dev/xdma0_c2h_0 -f ./data/output_bin/vggoutput_pack.bin -s $transferSize_rcv -c $transferCount &
		sleep 1
		$tool_path/dma_to_device -d /dev/xdma0_h2c_0 -f ./data/input_bin/tmp/tmp.bin -s $transferSize_send -c $transferCount &
		#echo "trans once end"
	fi
else
    echo "Info: No PCIe DMA stream channels were tested because no h2c/c2h pairs were found."
fi

# # Perform testing on the PCIe DMA core.
# testError=0
# if [ $isStreaming -eq 0 ]; then

# 	# Run the PCIe DMA memory mapped write read test
# 	./dma_memory_mapped_test.sh xdma0 $transferSize $transferCount $h2cChannels $c2hChannels
# 	returnVal=$?
# 	 if [ $returnVal -eq 1 ]; then
# 		testError=1
# 	fi

# else

# 	# Run the PCIe DMA streaming test
# 	channelPairs=$(($h2cChannels < $c2hChannels ? $h2cChannels : $c2hChannels))
# 	if [ $channelPairs -gt 0 ]; then
# 		./dma_streaming_test.sh $transferSize $transferCount $channelPairs
# 		returnVal=$?
# 		if [ $returnVal -eq 1 ]; then
# 			testError=1
# 		fi
# 	else
# 		echo "Info: No PCIe DMA stream channels were tested because no h2c/c2h pairs were found."
# 	fi

# fi

# # Exit with an error code if an error was found during testing
# if [ $testError -eq 1 ]; then
# 	echo "Error: Test completed with Errors."
# 	exit 1
# fi

# # Report all tests passed and exit
#echo "Info: All tests in run_tests.sh passed."

sleep 3
source /home/user/anaconda3/bin/activate
conda activate ytw
python ./result_count.py

exit 0
