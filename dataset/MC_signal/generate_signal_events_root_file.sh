#!/bin/bash

LOG_FILE="log_file_name.txt"
echo "Starting Monte Carlo generation at $(date)" | tee $LOG_FILE

NB_CORES=32

if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    SED_CMD="sed -i ''"
else
    # Linux and others
    SED_CMD="sed -i"
fi

# Array of NPcHW values to test
for value in -10.000000 -9.000000 -8.000000 -7.000000 -6.000000 -5.000000 -4.000000 -3.000000 -2.000000 -1.000000 -0.900000 -0.800000 -0.700000 -0.600000 -0.500000 -0.400000 -0.300000 -0.200000 -0.100000 0.000000 0.100000 0.200000 0.300000 0.400000 0.500000 0.600000 0.700000 0.800000 0.900000 1.000000 2.000000 3.000000 4.000000 5.000000 6.000000 7.000000 8.000000 9.000000 10.000000

do
    echo "----------------------------------------" | tee -a $LOG_FILE
    echo "Processing NPcHW value: $value" | tee -a $LOG_FILE
    
    # Make a backup of the original param_card.dat
    cp Cards/param_card.dat Cards/param_card.dat.bak
    
    # Modify param_card.dat with the new NPcHW value
    $SED_CMD "s/7 .* # chw/7 ${value}e+00 # chw/" Cards/param_card.dat
    
    # Verify the parameter was changed
    grep "# chw" Cards/param_card.dat | tee -a $LOG_FILE
    
    # Check if the value was properly set
    if ! grep -q "7 ${value}e+00 # chw" Cards/param_card.dat; then
        echo "ERROR: Failed to update chw parameter to $value!" | tee -a $LOG_FILE
        # Restore backup and continue with next value
        cp Cards/param_card.dat.bak Cards/param_card.dat
        continue
    fi
    
    echo "Generating events for cHW=$value using $NB_CORES cores..." | tee -a $LOG_FILE
    
    # Generate events with output displayed and logged
    # Using --nb_core parameter to enable multi-core processing
    ./bin/generate_events -f --nb_core=$NB_CORES events_cHW$value 2>&1 | tee -a $LOG_FILE
    
    # Check if the generation was successful
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "ERROR: Event generation failed for cHW=$value" | tee -a $LOG_FILE
    else
        echo "Successfully generated events for cHW=$value" | tee -a $LOG_FILE
    fi
done

echo "Monte Carlo generation completed at $(date)" | tee -a $LOG_FILE
