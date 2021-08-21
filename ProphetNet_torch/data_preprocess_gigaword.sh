echo start
fairseq-preprocess \
--user-dir prophetnet \
--task translation_prophetnet \
--source-lang src --target-lang tgt \
--testpref gigaword/prophetnet_tokenized/test \
--destdir gigaword/processed \
--srcdict vocab.txt \
--tgtdict vocab.txt \
--workers 5
#--trainpref gigaword/prophetnet_tokenized/train \
#--validpref gigaword/prophetnet_tokenized/dev \
echo end
while true
do
    command
done