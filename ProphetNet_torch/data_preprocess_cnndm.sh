echo start
fairseq-preprocess \
--user-dir ./prophetnet \
--task translation_prophetnet \
--source-lang src --target-lang tgt \
--trainpref cnndm/prophetnet_tokenized/train \
--validpref cnndm/prophetnet_tokenized/dev \
--testpref cnndm/prophetnet_tokenized/test \
--destdir cnndm/processed \
--srcdict ./vocab.txt \
--tgtdict ./vocab.txt \
--workers 5
#--workers 20

#echo '按下 <CTRL-D> 退出'
#echo -n '输入你最喜欢的网站名: '
#while read FILM
#do
#    echo "是的！$FILM 是一个好网站"
#done
echo end
while true
do
    command
done
