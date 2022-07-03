import re
import os
import sys
import json
import numpy as np
import random
import zipfile

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

def main(args):
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
                                                      limit_per_sentence=30)
            f.write(f'{ground_truth_sentence}\n')
            for sentence in similar_sentences:
                f.write(f'\t{sentence}\n')


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--esun_data_path',
                        type=str,
                        help='2022summer_train_data.zip file path')
    parser.add_argument('--extract_dir',
                        type=str,
                        help='the directory which 2022summer_train_data.zip extract to')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    sys.argv = [sys.argv[0]]
    sys.argv += ['--esun_data_path', '/home/hsiehpinghan/git/esun_ai_2022_summer_demo/model-spec/data/2022summer_train_data.zip']
    sys.argv += ['--extract_dir', '/home/hsiehpinghan/git/esun_ai_2022_summer_demo/model-spec/data']
    main(args=parse_args())