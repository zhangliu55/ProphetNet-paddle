echo start
SUFFIX=_ck9_pelt1.2_test_beam5
BEAM=5
LENPEN=1.2
#CHECK_POINT=cnndm/finetune_cnndm_checkpoints/checkpoint9.pt
CHECK_POINT=cnndm/finetune_cnndm_checkpoints/prophetnet_large_160G_cnndm_model.pt
OUTPUT_FILE=cnndm/output$SUFFIX.txt
SCORE_FILE=cnndm/score$SUFFIX.txt
#fairseq-generate cnndm/processed \
#--path $CHECK_POINT \
#--user-dir prophetnet \
#--task translation_prophetnet \
#--cpu \
#--skip-invalid-size-inputs-valid-test \
#--batch-size 3 \
#--gen-subset test \
#--beam $BEAM \
#--num-workers 4 \
#--min-len 45 \
#--max-len-b 110 \
#--no-repeat-ngram-size 3 \
#--lenpen $LENPEN 2>&1 > $OUTPUT_FILE
#
#grep ^H $OUTPUT_FILE | cut -c 3- | sort -n | cut -f3- | sed "s/ ##//g" > cnndm/sort_hypo$SUFFIX.txt
python cnndm/eval/postprocess_cnn_dm.py --generated cnndm/sort_hypo$SUFFIX.txt --golden cnndm/original_data/test.summary > $SCORE_FILE


echo end
while true
do
    command
done