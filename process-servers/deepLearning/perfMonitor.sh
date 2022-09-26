#!/bin/bash

# Global variables Section
#----------------------------------------------------------------------------
#Process target Name
processName="python"

#Measures per second 
measureFreq=5

#Output CSV File
outputCSV="./cpu_mem_perf.csv"
#----------------------------------------------------------------------------

# Methods
#----------------------------------------------------------------------------
#Waiting for 'processName' starts
function setPID() {
    if [ "$#" -lt 1 ]; then
        while true
        do
    	    pid=$(ps -A | grep $processName | tr -s ' ' | cut -f2 -d" ")
            if [ ! -z $pid ]; then
                break
            fi
        done
    else 
        pid=$1
    fi
}
#----------------------------------------------------------------------------

#Measurement Section
#----------------------------------------------------------------------------
#Get Total Memory 
totalMemory=$(cat /proc/meminfo | grep "MemTotal" | tr -s ' ' | cut -f2 -d' ')

#Set Measure Frequency
uSeconds=$(echo "scale=1; 1/$measureFreq" | bc)

echo "--------------------------------------------------------------------------"
echo "|               P E R F O R M A N C E    M E A T E R                     |"
echo "--------------------------------------------------------------------------"
echo "   * Starting Meater. Waiting to the target process '$processName'..."

#Get and Set PID
setPID

echo "   * Measuring process with PID $pid..."
echo "--------------------------------------------------------------------------"
echo 

echo "Time(s);Memory Usage(%);Memory Usage(Mb);CPU Usage(%)" > $outputCSV


#Start Monitorization
start=$(date +%s.%N)
while true
do
    measure=$(ps -p "$pid" -o %cpu,%mem,pid | grep "$pid" | tr -s ' ')
    mem=$( echo "$measure" | cut -f3 -d" ")
    if [ -z $mem ]; then 
        break
    fi
    cpu=$( echo "$measure" | cut -f2 -d" ")
    memUsed=$(echo "scale=1; (($mem * $totalMemory) / 100) / 1024" | bc)
    now=$(date +%s.%N)
    t=$(echo "$now - $start" | bc -l)
    echo -ne "[*] Time: $t(s) | PID: $pid | Memory Usage: $memUsed Mb ($mem%) | CPU Usage: ($cpu%)\r"
    echo "$t;$mem;$memUsed;$cpu" >> $outputCSV 
    sleep $uSeconds
done

#----------------------------------------------------------------------------
