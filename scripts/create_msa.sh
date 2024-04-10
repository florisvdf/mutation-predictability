#!/bin/bash

output_dir=$1
reference_path=$2
database_path=$3

bit_scores=$(seq 0.1 0.1 0.9)

for score in $bit_scores; do
  echo "Creating MSA for file ${reference_path} with bit score ${score}."
  jackhmmer --cpu 6 -T $score -A "${output_dir}/msa_${score}.sto" "${reference_path}" "${database_path}";
  echo "Reformatting MSA for to .a2m format.";
  esl-reformat a2m "${output_dir}/msa_${score}.sto" > "${output_dir}/msa_${score}.a2m"
done
