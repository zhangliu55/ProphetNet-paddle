from pytorch_transformers import BertTokenizer
import tqdm

def convert_cased2uncased(fin, fout):
    fin = open(fin, 'r', encoding='utf-8')
    fout = open(fout, 'w', encoding='utf-8')
    tok = BertTokenizer.from_pretrained('bert-base-uncased')
    for line in tqdm.tqdm(fin.readlines()):
        org = line.strip().replace(" ##", "")
        new = tok.tokenize(org)
        new_line = " ".join(new)
        fout.write('{}\n'.format(new_line))
# convert_cased2uncased('gigaword/unilm_tokenized/train.src', 'gigaword/prophetnet_tokenized/train.src')
# convert_cased2uncased('unilm_tokenized/test.src', 'prophetnet_tokenized/test.src')
convert_cased2uncased('unilm_tokenized/test.tgt', 'prophetnet_tokenized/test.tgt')
