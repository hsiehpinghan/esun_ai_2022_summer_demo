import re
import os
import sys
import json
import numpy as np
import random
import shutil
import zipfile
import urllib.request

from tqdm import tqdm
from argparse import ArgumentParser

def remove_not_chinese_char_or_space(text):
    text = re.sub(pattern='([^\u4e00-\u9fa5 ])+',
                  repl='',
                  string=text)
    return text

def is_sentence_similar(no_space_sentence, ground_truth_sentence):
    assert len(no_space_sentence) == len(ground_truth_sentence)
    is_char_equals = []
    for (char_0, char_1) in zip(list(no_space_sentence), list(ground_truth_sentence)):
        is_char_equals.append(1 if char_0 == char_1 else 0)
    if np.average(is_char_equals) > 0.5:
        return True
    else:
        return False

def get_wrong_word_dict(items):
    wrong_word_dict = {}
    for (i, item) in enumerate(items):
        id = item['id']
        ground_truth_sentence = remove_not_chinese_char_or_space(text=item['ground_truth_sentence'])
        sentence_list = set()
        for sentence in item['sentence_list']:
            sentence = remove_not_chinese_char_or_space(text=sentence)
            sentence_list.add(sentence)
        for sentence in sentence_list:
            no_space_sentence = sentence.replace(' ', '')
            if len(no_space_sentence) != len(ground_truth_sentence):
                continue
            if is_sentence_similar(no_space_sentence, ground_truth_sentence) == False:
                continue
            index = 0
            for word in sentence.split(' '):
                key = ground_truth_sentence[index:(index+len(word))]
                if key != word:
                    if key not in wrong_word_dict:
                        wrong_word_dict[key] = set()
                    wrong_word_dict[key].add(word)
                index += len(word)
    return wrong_word_dict

def get_similar_sentence(ground_truth_sentence, words, wrong_word_dict):
    word_replace_amt = (len(ground_truth_sentence) // 10) + 1
    random.shuffle(words)
    sentence_tmp = ground_truth_sentence
    for word in words:
        if word_replace_amt <= 0:
            break
        if word in sentence_tmp:
            wrong_word = random.choice(list(wrong_word_dict[word]))
            replace_type = random.randint(0, 2)
            if replace_type == 0:
                # replace first from beging
                sentence_tmp = re.sub(word, wrong_word, sentence_tmp, 1)
            elif replace_type == 1:
                word_rev = word[::-1]
                wrong_word_rev = wrong_word[::-1]
                sentence_tmp_rev = sentence_tmp[::-1]
                sentence_tmp_rev = re.sub(word_rev, wrong_word_rev, sentence_tmp_rev, 1)
                sentence_tmp = sentence_tmp_rev[::-1]
            elif replace_type == 2:
                sentence_tmp = sentence_tmp.replace(word, wrong_word)
            else:
                raise Exception(f'replace_type({replace_type}) not implement !!!')
            word_replace_amt -= 1
    return sentence_tmp

def get_similar_sentences(ground_truth_sentence, sentence_list, words, wrong_word_dict, limit_per_sentence, is_add_ground_truth_sentence=True):
    similar_sentences = set()
    for _ in range(limit_per_sentence * 3):
        similar_sentence = get_similar_sentence(ground_truth_sentence=ground_truth_sentence,
                                                words=words,
                                                wrong_word_dict=wrong_word_dict)
        similar_sentences.add(similar_sentence)
        if len(similar_sentences) >= limit_per_sentence:
            break
    base_sentence_list = set()
    for sentence in sentence_list:
        sentence = remove_not_chinese_char_or_space(text=sentence)
        if len(sentence) != len(ground_truth_sentence):
            continue
        base_sentence_list.add(sentence)
    if is_add_ground_truth_sentence is True:
        base_sentence_list.add(ground_truth_sentence)
    for similar_sentence in similar_sentences:
        if len(base_sentence_list) >= limit_per_sentence:
            break
        base_sentence_list.add(similar_sentence)
    return base_sentence_list

def preprocess_esun_data(args):
    with zipfile.ZipFile(file=args.esun_data_path,
                         mode='r') as f:
        f.extractall(args.extract_dir)
    with open(file=os.path.join(args.extract_dir, 'train_all.json'),
              mode='r',
              encoding='utf-8') as f:
        items = json.load(fp=f)
    wrong_word_dict = get_wrong_word_dict(items=items)
    with open(file=os.path.join(args.extract_dir, 'esun_ai_2022_summer_20220415.txt'),
              mode='w',
              encoding='utf-8-sig') as f:
        for (i, item) in tqdm(enumerate(items),
                              total=len(items)):
            id = i
            ground_truth_sentence = remove_not_chinese_char_or_space(text=item['ground_truth_sentence'])
            words = []
            for word in wrong_word_dict:
                if word in ground_truth_sentence:
                    words.append(word)
            similar_sentences = get_similar_sentences(ground_truth_sentence=ground_truth_sentence,
                                                      sentence_list=item['sentence_list'],
                                                      words=words,
                                                      wrong_word_dict=wrong_word_dict,
                                                      limit_per_sentence=args.limit_per_sentence)
            f.write(f'{ground_truth_sentence}\n')
            for sentence in similar_sentences:
                f.write(f'\t{sentence}\n')

def get_cns_to_bopomofos(args):
    cns_to_bopomofos = {}
    with open(file=os.path.join(args.extract_dir, 'Open_Data/Properties', 'CNS_phonetic.txt')) as f:
        for (i, line) in enumerate(f):
            split_result = line.strip().split('\t')
            if len(split_result) <= 1:
                continue
            (cns, bopomofo) = split_result
            if cns not in cns_to_bopomofos:
                cns_to_bopomofos[cns] = [bopomofo]
            else:
                cns_to_bopomofos[cns].append(bopomofo)
    size_tmp = 0
    for cns in cns_to_bopomofos:
        size_tmp += len(cns_to_bopomofos[cns])
    return cns_to_bopomofos

def get_char_to_bopomofos(args):
    char_to_bopomofos = {}
    cns2unicode_file = [os.path.join(args.extract_dir, 'Open_Data/MapingTables/Unicode', 'CNS2UNICODE_Unicode BMP.txt'),
                        os.path.join(args.extract_dir, 'Open_Data/MapingTables/Unicode', 'CNS2UNICODE_Unicode 2.txt'),
                        os.path.join(args.extract_dir, 'Open_Data/MapingTables/Unicode', 'CNS2UNICODE_Unicode 15.txt')]
    cns_to_bopomofos = get_cns_to_bopomofos(args=args)
    for cns2unicode_file in cns2unicode_file:
        with open(file=cns2unicode_file) as f:
            for line in f:
                (cns, utf8) = line.strip().split('\t')
                char_ = chr(int(utf8,16))
                if cns not in cns_to_bopomofos:
                    continue
                bopomofos = cns_to_bopomofos[cns]
                char_to_bopomofos[char_] = bopomofos
    return char_to_bopomofos

def get_bopomofo_to_bert_ids(args,
                             char_to_bopomofos):
    bopomofo_to_bert_ids = {}
    with open(file=os.path.join(args.extract_dir, 'vocab.txt')) as f:
        for (i, line) in enumerate(f):
            if line.startswith('##'):
                continue
            else:
                char_ = line[0]
            if char_ not in char_to_bopomofos:
                continue
            bopomofos = char_to_bopomofos[char_]
            for bopomofo in bopomofos:
                if bopomofo not in bopomofo_to_bert_ids:
                    bopomofo_to_bert_ids[bopomofo] = [i]
                else:
                    bopomofo_to_bert_ids[bopomofo].append(i)
    return bopomofo_to_bert_ids

def update_similarity_bert_ids(similarity_bert_ids, bopomofo_to_bert_ids, bopomofo_modified):
    if 'ㄥ' in bopomofo_modified:
        bopomofo_tmp = bopomofo_modified.replace('ㄥ', 'ㄣ')
        #print(bopomofo_modified, bopomofo_tmp)
        if bopomofo_tmp in bopomofo_to_bert_ids:
            bert_ids = bopomofo_to_bert_ids[bopomofo_tmp]
            similarity_bert_ids.update(bert_ids)
    if 'ㄣ' in bopomofo_modified:
        bopomofo_tmp = bopomofo_modified.replace('ㄣ', 'ㄥ')
        #print(bopomofo_modified, bopomofo_tmp)
        if bopomofo_tmp in bopomofo_to_bert_ids:
            bert_ids = bopomofo_to_bert_ids[bopomofo_tmp]
            similarity_bert_ids.update(bert_ids)
    if 'ㄌ' in bopomofo_modified:
        bopomofo_tmp = bopomofo_modified.replace('ㄌ', 'ㄖ')
        #print(bopomofo_modified, bopomofo_tmp)
        if bopomofo_tmp in bopomofo_to_bert_ids:
            bert_ids = bopomofo_to_bert_ids[bopomofo_tmp]
            similarity_bert_ids.update(bert_ids)
    if 'ㄖ' in bopomofo_modified:
        bopomofo_tmp = bopomofo_modified.replace('ㄖ', 'ㄌ')
        #print(bopomofo_modified, bopomofo_tmp)
        if bopomofo_tmp in bopomofo_to_bert_ids:
            bert_ids = bopomofo_to_bert_ids[bopomofo_tmp]
            similarity_bert_ids.update(bert_ids)
    if 'ㄓ' in bopomofo_modified:
        bopomofo_tmp = bopomofo_modified.replace('ㄓ', 'ㄗ')
        #print(bopomofo_modified, bopomofo_tmp)
        if bopomofo_tmp in bopomofo_to_bert_ids:
            bert_ids = bopomofo_to_bert_ids[bopomofo_tmp]
            similarity_bert_ids.update(bert_ids)
    if 'ㄗ' in bopomofo_modified:
        bopomofo_tmp = bopomofo_modified.replace('ㄗ', 'ㄓ')
        #print(bopomofo_modified, bopomofo_tmp)
        if bopomofo_tmp in bopomofo_to_bert_ids:
            bert_ids = bopomofo_to_bert_ids[bopomofo_tmp]
            similarity_bert_ids.update(bert_ids)
    bert_ids = bopomofo_to_bert_ids[bopomofo_modified]
    similarity_bert_ids.update(bert_ids)

def get_char_to_similarity_bert_ids(args):
    char_to_similarity_bert_ids = {}
    char_to_bopomofos = get_char_to_bopomofos(args=args)
    bopomofo_to_bert_ids = get_bopomofo_to_bert_ids(args=args,
                                                    char_to_bopomofos=char_to_bopomofos)
    for (i, char_) in enumerate(char_to_bopomofos):
        similarity_bert_ids = set()
        for bopomofo in char_to_bopomofos[char_]:
            if bopomofo not in bopomofo_to_bert_ids:
                continue
            bopomofo_tmp = None
            if bopomofo[0] == '˙':
                bopomofo_tmp = bopomofo[1:]
            elif bopomofo[-1] in ['ˊ', 'ˇ', 'ˋ']:
                bopomofo_tmp = bopomofo[:-1]
            else:
                bopomofo_tmp = bopomofo
            bopomofo_modified = '˙' + bopomofo_tmp
            if bopomofo_modified in bopomofo_to_bert_ids:
                update_similarity_bert_ids(similarity_bert_ids=similarity_bert_ids,
                                           bopomofo_to_bert_ids=bopomofo_to_bert_ids,
                                           bopomofo_modified=bopomofo_modified)
            for sound in ['', 'ˊ', 'ˇ', 'ˋ']:
                bopomofo_modified = bopomofo_tmp + sound
                if bopomofo_modified in bopomofo_to_bert_ids:
                    update_similarity_bert_ids(similarity_bert_ids=similarity_bert_ids,
                                            bopomofo_to_bert_ids=bopomofo_to_bert_ids,
                                            bopomofo_modified=bopomofo_modified)
        char_to_similarity_bert_ids[char_] = list(similarity_bert_ids)
    return char_to_similarity_bert_ids

def download(url, dir_path, file_name=None):
    with urllib.request.urlopen(url=url) as response:
        if file_name is None:
            resource_name = url.split('/')[-1].strip()
            if resource_name == '':
                raise Exception(f'url({url}) has no resource_name !!!')
            file_name = resource_name
        file_path = os.path.join(dir_path, file_name)
        os.makedirs(name=dir_path,
                    exist_ok=True)
        with open(file=file_path,
                    mode='wb') as f:
            shutil.copyfileobj(fsrc=response,
                                fdst=f)
        return file_path
        
def download_cns11643(args):
    # download
    open_data_zip = os.path.join(args.extract_dir, 'Open_Data.zip')
    if os.path.exists(path=open_data_zip) == False:
        url = 'https://www.cns11643.gov.tw/AIDB/Open_Data.zip'
        print(f'download {url} to {args.extract_dir}')
        file_path = download(url=url,
                             dir_path=args.extract_dir)
    # unzip
    print(f'extract {open_data_zip} to {args.extract_dir}')
    with zipfile.ZipFile(file=open_data_zip,
                         mode='r') as zf:
        zf.extractall(path=args.extract_dir)

def download_vocab(args):
    # download
    url = 'https://huggingface.co/shibing624/macbert4csc-base-chinese/resolve/main/vocab.txt'
    print(f'download {url} to {args.extract_dir}')
    file_path = download(url=url,
                         dir_path=args.extract_dir)
  
def preprocess_char_to_similarity_bert_ids(args):
    download_cns11643(args=args)
    download_vocab(args=args)
    char_to_similarity_bert_ids = get_char_to_similarity_bert_ids(args=args)
    with open(file=args.char_to_similarity_bert_ids_file_path,
            mode='w') as f:
        json.dump(obj=char_to_similarity_bert_ids,
                  fp=f,
                  ensure_ascii=False,
                  indent=4)

def main(args):
    preprocess_esun_data(args=args)
    preprocess_char_to_similarity_bert_ids(args=args)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--esun_data_path',
                        type=str,
                        help='2022summer_train_data.zip file path')
    parser.add_argument('--extract_dir',
                        type=str,
                        help='the directory which 2022summer_train_data.zip extract to')
    parser.add_argument('--limit_per_sentence',
                        type=int,
                        help='max similar sentences size')
    parser.add_argument('--char_to_similarity_bert_ids_file_path',
                        type=str,
                        help='where the char_to_similarity_bert_ids.json write to')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    """
    sys.argv = [sys.argv[0]]
    sys.argv += ['--esun_data_path', '/home/hsiehpinghan/git/esun_ai_2022_summer_demo/model-spec/data/2022summer_train_data.zip']
    sys.argv += ['--extract_dir', '/home/hsiehpinghan/git/esun_ai_2022_summer_demo/model-spec/data']
    sys.argv += ['--limit_per_sentence', '3']
    sys.argv += ['--char_to_similarity_bert_ids_file_path', '/home/hsiehpinghan/git/esun_ai_2022_summer_demo/model-spec/data/char_to_similarity_bert_ids.json']    
    """
    main(args=parse_args())