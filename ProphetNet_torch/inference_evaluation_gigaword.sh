echo start
SUFFIX=_ck7_pelt1.0_test_beam4
BEAM=4
LENPEN=1.0
CHECK_POINT=gigaword/finetune_gigaword_checkpoints/checkpoint7.pt
OUTPUT_FILE=gigaword/output$SUFFIX.txt
SCORE_FILE=gigaword/score$SUFFIX.txt
#
#fairseq-generate gigaword/processed \
#--path $CHECK_POINT \
#--user-dir prophetnet \
#--task translation_prophetnet \
#--batch-size 80 \
#--gen-subset test \
#--beam $BEAM \
#--num-workers 4 \
#--lenpen $LENPEN 2>&1 > $OUTPUT_FILE
#grep ^H $OUTPUT_FILE | cut -c 3- | sort -n | cut -f3- | sed "s/ ##//g" > gigaword/sort_hypo$SUFFIX.txt

#git clone https://github.com/andersjo/pyrouge.git
#set pyrouge_set_rouge_path=E:\移动文件资料\百度论文复现赛\ProphetNet-master\ProphetNet_En\pyrouge\tools\ROUGE-1.5.5
set pyrouge_set_rouge_path=E:\pyrouge\tools\ROUGE-1.5.5
chmod +x E:/pyrouge/tools/ROUGE-1.5.5/ROUGE-1.5.5.pl
python gigaword/eval/eval.py --pred gigaword/sort_hypo_ck7_pelt1.0_test_beam4.txt --gold gigaword/original_data/test.tgt.txt --perl > $SCORE_FILE

echo end
while true
do
    command
done