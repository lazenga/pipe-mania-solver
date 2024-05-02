#!/bin/bash

green='\033[0;32m'
red='\033[0;31m'
end_color='\033[0m'

temp_dir="temp-output"

if [ -d "$temp_dir" ]; then
    rm -r "$temp_dir"
fi

mkdir "$temp_dir"

for i in {1..19}; do
    rm -r __pycache__
    # Input and expected output files
    tests_path="tests"
    input_file="$tests_path/test-$i.txt"
    expected_output_file="$tests_path/test-$i.out"
    temp_output_file="$temp_dir/test-$i.out"

    # Execute and get the real time
    exec_time=$(time (python3 pipe.py < "$input_file" > "$temp_output_file") 2>&1)
    exec_time=$(echo "$exec_time" | grep real | awk '{print $2}')

    # Compare the output with the expected output
    echo "---------------"
    if [ "$(cat "$temp_output_file")" = "$(cat "$expected_output_file")" ]; then
        echo -e "${green}Test $i: PASSED${end_color} (Execution Time: $exec_time)"
    else
        echo -e "${red}Test $i: FAILED${end_color} (Execution Time: $exec_time)"
        echo -e "${red}Expected output:${end_color}"
        cat "$expected_output_file"
        echo -e "${red}Actual output:${end_color}"
        cat "$temp_output_file"
    fi
done

echo "---------------"
