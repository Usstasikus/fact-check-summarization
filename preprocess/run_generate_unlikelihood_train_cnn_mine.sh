#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT

# split the training input into smaller sub files to avoid OOM errors:
#split -l 10000 /home/svdon/data/CNN/stories_files/processed-data/train.source /home/svdon/data/CNN/stories_files/processed-data/train_data_split/train.source.split
#split -l 10000 /home/svdon/data/CNN/stories_files/processed-data/train.target /home/svdon/data/CNN/stories_files/processed-data/train_data_split/train.target.split

# 0. Obtain the MLE summarization baseline by fine-tuning the BART model
# See README for standard BART fine-tuning

# 1. Use the current summarization model to sample summaries on the training data.
for i in {1..60}
do
  echo "Running $i times"
  python sm_inference_asum.py --task gen_summary --base_dir /home/svdon/data/CNN/stories_files/processed-data/train_data_split --input_file train.source.split --num_workers 1 --bsz 30 --sampling True --sampling_topk 50 --beam 6 --max_len 60 --min_len 10 --checkpoint_dir /home/svdon/data/checkpoints/bart_xsum_cnndm --ckp_file checkpoint8.pt --bin_dir /home/svdon/data/CNN/stories_files/processed-data/data_bin --output_dir /home/svdon/data/CNN/stories_files/processed-data/train_data_split/mine_sm_inference_asum_model_gen_summaries
done

# 2. Generate question and answer pairs from summaries
for i in {1..60}
do
  echo "Running $i times"
  python sm_inference_asum.py --task gen_qa --base_dir /home/svdon/data/CNN/stories_files/processed-data/train_data_split --source_dir mine_sm_inference_asum_model_gen_summaries --output_dir /home/svdon/data/CNN/stories_files/processed-data/train_data_split/mine_sm_inference_asum_model_gen_qa --num_workers 1 --bsz 10 --beam 60 --max_len 60 --min_len 8 --checkpoint_dir /home/svdon/downloads/qagen --ckp_file checkpoint2.pt --bin_dir /home/svdon/data/CNN/stories_files/processed-data/data_bin --diverse_beam_groups 60 --diverse_beam_strength 0.5 --input_file train.source.split*.hypo --return_token_scores True
done

# 3. Filter the generated question and answer for high quality pairs
python evaluate_hypo.py --mode filter_qas_dataset_lm_score --base_dir /home/svdon/data/CNN/stories_files/processed-data/train_data_split --sub_dir mine_sm_inference_asum_model_gen_qa --pattern train.target.split*.qas

# 4. Evaluate the generated question and answer pairs using the source document as input
for i in {1..60}
do
  echo "Running $i times"
  python sm_inference_asum.py --task qa_eval --base_dir /home/svdon/data/CNN/stories_files/processed-data/train_data_split --output_dir /home/svdon/data/CNN/stories_files/processed-data/train_data_split/mine_sm_inference_asum_model_gen_qa_eval --num_workers 1 --bsz 60 --checkpoint_dir /home/svdon/downloads/qagen --ckp_file checkpoint2.pt --bin_dir /home/svdon/data/CNN/stories_files/processed-data/data_bin --qas_dir /home/svdon/data/CNN/stories_files/processed-data/train_data_split/mine_sm_inference_asum_model_gen_qa --source_file train.source.split* --target_file train.target.split* --input_file *.qas_filtered --prepend_target False
done

# 5. compute the lm scores for the ground truth training summaries
python evaluate_hypo.py --mode select_unlikelihood_hypos_lm_score --base_dir /home/svdon/data/CNN/stories_files/processed-data/train_data_split --sub_dir mine_sm_inference_asum_model_gen_qa_eval --pattern train.*.source_eval_noprepend

# In order to apply CONSEQ, we need to evaluate the QUALS of the ground truth summaries of the training set as well:

# 1a. Convert the ground truth target to jsonl format:
python evaluate_hypo.py --mode convert_hypo_to_json --base_dir /home/svdon/data/CNN/stories_files/processed-data/ --sub_dir train_data_split/ --split train --pattern .target.split*

# 2a. Generate question and answer pairs from summaries
for i in {1..60}
do
  echo "Running $i times"
  python sm_inference_asum.py --task gen_qa --base_dir /home/svdon/data/CNN/stories_files/processed-data --source_dir train_data_split/ --output_dir /home/svdon/data/CNN/stories_files/processed-data/train_data_split/mine_sm_inference_asum_target_gen_qa --num_workers 1 --bsz 8 --beam 60 --max_len 60 --min_len 8 --checkpoint_dir /home/svdon/downloads/qagen --ckp_file checkpoint2.pt --bin_dir /home/svdon/data/CNN/stories_files/processed-data/data_bin/data_bin --diverse_beam_groups 60 --diverse_beam_strength 0.5 --input_file train.target.split*.hypo --return_token_scores True --batch_lines True
done

# 3a. Filter the generated question and answer for high quality pairs
python evaluate_hypo.py --mode filter_qas_dataset_lm_score --base_dir /home/svdon/data/CNN/stories_files/processed-data/train_data_split --sub_dir mine_sm_inference_asum_target_gen_qa --pattern train.target.split*.qas

# 4a. Evaluate the generated question and answer pairs using the source document as input
for i in {1..60}
do
  echo "Running $i times"
  python sm_inference_asum.py --task qa_eval --base_dir /home/svdon/data/CNN/stories_files/processed-data/train_data_split --output_dir /home/svdon/data/CNN/stories_files/processed-data/train_data_split/mine_sm_inference_asum_target_gen_qa_eval --num_workers 1 --bsz 60 --checkpoint_dir /home/svdon/downloads/qagen --ckp_file checkpoint2.pt --bin_dir /home/svdon/data/CNN/stories_files/processed-data/data_bin/data_bin --qas_dir /home/svdon/data/CNN/stories_files/processed-data/train_data_split/mine_sm_inference_asum_target_gen_qa --source_file train.source.split* --target_file train.target.split* --input_file *.qas_filtered --prepend_target False
done

# 5a. compute the lm scores for the ground truth training summaries
python evaluate_hypo.py --mode select_unlikelihood_hypos_lm_score --base_dir /home/svdon/data/CNN/stories_files/processed-data/train_data_split --sub_dir mine_sm_inference_asum_target_gen_qa_eval --pattern train.*.source_eval_noprepend

# 6. make positive and negative training set for contrastive learning
ratio=0.3
targetRatio=0.3
ratio100=30
targetRatio100=30
type=lm
echo "$type-$ratio-$targetRatio"

# running the below command will create a sub-directory called $type-$ratio100-$targetRatio100 that contains the training data for contrastive learning (train.source for source documents; train.target for positive summaries and train.untarget for negative summaries.)
python evaluate_hypo.py --mode make_unlikelihood_dataset --base_dir /home/svdon/data/CNN/stories_files/processed-data/train_data_split --sub_dir mine_sm_inference_asum_model_gen_qa_eval --pattern train.source.split*.source_eval_noprepend --unlike_select_ratio $ratio --score_type $type --target_select_ratio $targetRatio --target_index_file /home/svdon/data/CNN/stories_files/processed-data/train_data_split/mine_sm_inference_asum_target_gen_qa_eval/untarget.index --metric_choice eval_ns-ns

# 7. Binarize the data for training
python data_prepro_clean.py --mode bpe_binarize --input_dir /home/svdon/data/CNN/stories_files/processed-data/train_data_split/mine_sm_inference_asum_model_gen_qa_eval/"$type"-"$ratio100"-"$targetRatio100" --tokenizer_dir /home/svdon/data/BPE --no_val
python data_prepro_clean.py --mode binarize_untarget --input_dir /home/svdon/data/CNN/stories_files/processed-data/train_data_split/mine_sm_inference_asum_model_gen_qa_eval/"$type"-"$ratio100"-"$targetRatio100" --tokenizer_dir /home/svdon/data/BPE --no_val

# 8. Re-use the binarized source inputs for positive and negative examples
ln -s /home/svdon/data/CNN/stories_files/processed-data/train_data_split/mine_sm_inference_asum_model_gen_qa_eval/"$type"-"$ratio100"-"$targetRatio100"/data_bin/train.source-target.source.bin /home/svdon/data/CNN/stories_files/processed-data/train_data_split/mine_sm_inference_asum_model_gen_qa_eval/"$type"-"$ratio100"-"$targetRatio100"/data_bin/train.source-untarget.source.bin
ln -s /home/svdon/data/CNN/stories_files/processed-data/train_data_split/mine_sm_inference_asum_model_gen_qa_eval/"$type"-"$ratio100"-"$targetRatio100"/data_bin/train.source-target.source.idx /home/svdon/data/CNN/stories_files/processed-data/train_data_split/mine_sm_inference_asum_model_gen_qa_eval/"$type"-"$ratio100"-"$targetRatio100"/data_bin/train.source-untarget.source.idx