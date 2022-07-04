import re
import sys
import json
import torch
import numpy as np
import torch.nn as nn

from typing import Tuple
from typing import Union
from typing import Optional
from argparse import ArgumentParser
from collections import defaultdict
from transformers import AutoTokenizer
from transformers import BertForMaskedLM
from transformers.modeling_outputs import MaskedLMOutput

class EnsembleModel(nn.Module):
    def __init__(self, pretrained_model_name_or_paths, device):
        super().__init__()
        models = []
        for pretrained_model_name_or_path in pretrained_model_name_or_paths:
            model = BertForMaskedLM.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path).to(device)
            model.eval()
            models.append(model)
        self._models = models

    def forward(self,
                input_ids: Optional[torch.Tensor],
                attention_mask: Optional[torch.Tensor],
                token_type_ids: Optional[torch.Tensor],
                correction_labels: Optional[torch.Tensor] = None,
                detection_labels: Optional[torch.Tensor] = None,
                output_hidden_states: Optional[bool] = True,
                return_dict: Optional[bool] = True,
                position_ids: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                encoder_hidden_states: Optional[torch.Tensor] = None,
                encoder_attention_mask: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = None) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        sub_probs_list = []
        for model in self._models:
            outputs = model.forward(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids,
                                    position_ids=position_ids,
                                    head_mask=head_mask,
                                    inputs_embeds=inputs_embeds,
                                    encoder_hidden_states=encoder_hidden_states,
                                    encoder_attention_mask=encoder_attention_mask,
                                    labels=correction_labels,
                                    output_attentions=output_attentions,
                                    output_hidden_states=output_hidden_states,
                                    return_dict=return_dict)
            sub_probs = torch.softmax(outputs.logits,
                                      dim=-1)
            sub_probs_list.append(sub_probs)
        probs = torch.stack(sub_probs_list,
                            dim=0)\
            .mean(dim=0)
        return probs

def is_char_equal(char_0, char_1):
    return 1 if char_0 == char_1 else 0
    
def split_list_if_different_too_much(sentences):
    result = []
    sentences_tmp = sentences.copy()
    while len(sentences_tmp) > 0:
        main_ele = sentences_tmp.pop(-1)
        result.append([main_ele])
        is_char_equals = []
        for i in range(len(sentences_tmp)-1, -1, -1):
            assert len(main_ele) == len(sentences_tmp[i])
            for (char_0, char_1) in zip(list(main_ele), list(sentences_tmp[i])):
                is_char_equals.append(is_char_equal(char_0=char_0,
                                                            char_1=char_1))
            if np.average(is_char_equals) > 0.5:
                result[-1].append(sentences_tmp.pop(i))
    return result
    
def get_sentences_list_similar(sentence_list):
    """
    將相同長度且字相似比例>50%的sentence歸為一組。
    """
    sentences_dict = defaultdict(list)
    for sentence in sentence_list:
        sentences_dict[len(sentence)].append(sentence)
    sentences_list = sentences_dict.values()
    # filter len >= 1
    sentences_list = [
        sentences for sentences in sentences_list if len(sentences) >= 1]
    # split list if different too much
    sentences_list_similar = []
    for sentences in sentences_list:
        sentences_list_similar += split_list_if_different_too_much(
            sentences=sentences)
    # filter len >= 1
    sentences_list_similar = [
        sentences for sentences in sentences_list_similar if len(sentences) >= 1]
    return sentences_list_similar
    
def get_chinese_only_sentences(sentences):
    chinese_only_sentences = []
    for sentence in sentences:
        chinese_only_sentence = re.sub(pattern='([^\u4e00-\u9fa5])+',
                                        repl='',
                                        string=sentence)
        chinese_only_sentences.append(chinese_only_sentence)
    return chinese_only_sentences

def get_sentences_similar(sentences_list_similar):
    """
    本函式本來有額外功能，
    但現在已經不使用該功能，
    所以目前只是單純做flatten。
    """
    return [sentence_similar for sentences_similar in sentences_list_similar for sentence_similar in sentences_similar]

def get_char_to_similarity_bert_ids(char_to_similarity_bert_ids_file_path):
    """
    取得每個字相似讀音的字屬於那些bert的embedding id
    """
    with open(file=char_to_similarity_bert_ids_file_path,
              mode='r') as f:
        char_to_similarity_bert_ids = json.load(fp=f)
    return char_to_similarity_bert_ids

def get_similarity_bert_ids_list(char_to_similarity_bert_ids_file_path, sentences_similar):
    """
    取得每個sentence相似讀音的字屬於那些bert的embedding id
    """
    similarity_bert_ids_list = [{} for _ in sentences_similar]
    char_to_similarity_bert_ids = get_char_to_similarity_bert_ids(char_to_similarity_bert_ids_file_path=char_to_similarity_bert_ids_file_path)
    for (sentence_index, sentence_similar) in enumerate(sentences_similar):
        for (char_index, diff_char) in enumerate(sentence_similar):
            if diff_char not in char_to_similarity_bert_ids:
                similarity_bert_ids = [100]
            else:
                similarity_bert_ids = char_to_similarity_bert_ids[diff_char]
                if len(similarity_bert_ids) <= 0:
                    similarity_bert_ids = [100]
            similarity_bert_ids_list[sentence_index][char_index] = similarity_bert_ids
    return similarity_bert_ids_list

def get_most_possible_sentences(device, model, tokenizer, sentences_similar, similarity_bert_ids_list):
    """
    推論出一段句子中每個字的位置替換成其他字的機率後，
    以input句子中每個字的發音決定最後可能的替代字及其機率，
    例如：
        輸入：可能導致不是泡沫在見
        原始輸出：可能導致房市泡沫再現 (這邊以字代替logits方便說明)
        修正後輸出：可能導致股市泡沫再現 (因為只會選logits結果中與”不”的音相近的字，因此最後”股”是在這個限制條件下機率最大的字)
    最後將sentence_list中各sentence推論的結果中各字的機率做平均，
    取分數最高的sentence視為本次推論的結果。
    """
    with torch.no_grad():
        probs_list = model(**tokenizer(sentences_similar,
                                       padding=True,
                                       return_tensors='pt').to(device))
    result = []
    assert len(probs_list) == len(
        sentences_similar), f'{len(probs_list)} / {len(sentences_similar)}'
    assert len(probs_list) == len(
        similarity_bert_ids_list), f'{len(probs_list)} / {len(similarity_bert_ids_list)}'
    for (probs, sentence_similar, similarity_bert_ids) in zip(probs_list, sentences_similar, similarity_bert_ids_list):
        assert len(sentence_similar) <= (
            len(probs)-2), f'{len(sentence_similar)} / {len(probs)-2}'
        chars = []
        ps = []
        for (i, char_) in enumerate(sentence_similar):
            if i not in similarity_bert_ids:
                chars.append(char_)
            else:
                similarity_bert_id = similarity_bert_ids[i][torch.argmax(
                    probs[[i+1], similarity_bert_ids[i]])]
                char_ = tokenizer.decode(similarity_bert_id)
                ps.append(probs[[i+1], similarity_bert_id].cpu().numpy())
                chars.append(char_)
        result.append((''.join(chars), np.average(ps)))
    return sorted(result,
                    key=lambda x: x[1],
                    reverse=True)

def get_predict_sentence(device, model, tokenizer, sentences_similar, similarity_bert_ids_list):
    most_possible_sentences = get_most_possible_sentences(device=device,
                                                          model=model,
                                                          tokenizer=tokenizer,
                                                          sentences_similar=sentences_similar,
                                                          similarity_bert_ids_list=similarity_bert_ids_list)
    return most_possible_sentences[0][0]

def get_tokenizer(checkpoints):
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=checkpoints[0])
    return tokenizer
    
def get_model(checkpoints, device):
    """
    將5個模型ensemble成最終的模型。
    """
    model = EnsembleModel(pretrained_model_name_or_paths=checkpoints,
                          device=device)
    return model

def get_answer(tokenizer, model, request, char_to_similarity_bert_ids_file_path, device):
    sentence_list = request['sentence_list']
    chinese_only_sentences = get_chinese_only_sentences(sentences=sentence_list)
    sentences_list_similar = get_sentences_list_similar(
        sentence_list=chinese_only_sentences)
    sentences_similar = get_sentences_similar(
        sentences_list_similar=sentences_list_similar)
    similarity_bert_ids_list = get_similarity_bert_ids_list(char_to_similarity_bert_ids_file_path,
                                                            sentences_similar=sentences_similar)
    answer = get_predict_sentence(device=device,
                                  model=model,
                                  tokenizer=tokenizer,
                                  sentences_similar=sentences_similar,
                                  similarity_bert_ids_list=similarity_bert_ids_list)
    return answer

def main(args):
    tokenizer = get_tokenizer(checkpoints=args.checkpoints)
    model = get_model(checkpoints=args.checkpoints,
                      device=args.device)
    answer = get_answer(tokenizer=tokenizer,
                        model=model,
                        request=json.loads(args.request),
                        char_to_similarity_bert_ids_file_path=args.char_to_similarity_bert_ids_file_path,
                        device=args.device)
    print(f'inference result: {answer}')

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--request',
                        type=str,
                        help='request')
    parser.add_argument('--char_to_similarity_bert_ids_file_path',
                        type=str,
                        help='where the char_to_similarity_bert_ids.json write to')
    parser.add_argument('--checkpoints',
                        type=str,
                        nargs='+',
                        help='checkpoints to be ensembled')
    parser.add_argument('--device',
                        type=str,
                        help='cpu or gpu')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    """
    sys.argv = [sys.argv[0]]
    sys.argv += ['--request', '{"esun_uuid": "adefb7e8d9268b972b95b6fa53db93780b6b22fbf", "esun_timestamp": 1590493849, "sentence_list": ["喂 你好 密碼 我 要 進去", "喂 你好 密碼 哇 進去", "喂 你好 密碼 的 話 進去", "喂 您好 密碼 我 要 進去", "喂 你好 密碼 無法 進去", "喂 你好 密碼 waa 進去", "喂 你好 密碼 while 進去", "喂 你好 密碼 文化 進去", "喂 你好 密碼 挖 進去", "喂 您好 密碼 哇 進去"], "phoneme_sequence_list": ["w eI4 n i:3 x aU4 m i:4 m A:3 w O:3 j aU1 ts6 j ax n4 ts6_h y4", "w eI4 n i:3 x aU4 m i:4 m A:3 w A:1 ts6 j ax n4 ts6_h y4", "w eI4 n i:3 x aU4 m i:4 m A:3 t ax5 x w A:4 ts6 j ax n4 ts6_h y4", "w eI4 n j ax n2 x aU4 m i:4 m A:3 w O:3 j aU1 ts6 j ax n4 ts6_h y4", "w eI4 n i:3 x aU4 m i:4 m A:3 u:2 f A:4 ts6 j ax n4 ts6_h y4", "w eI4 n i:3 x aU4 m i:4 m A:3 W AA1 ts6 j ax n4 ts6_h y4", "w eI4 n i:3 x aU4 m i:4 m A:3 W AY1 L ts6 j ax n4 ts6_h y4", "w eI4 n i:3 x aU4 m i:4 m A:3 w ax n2 x w A:4 ts6 j ax n4 ts6_h y4", "w eI4 n j ax n2 x aU4 m i:4 m A:3 w A:1 ts6 j ax n4 ts6_h y4", "w eI4 n i:3 x aU4 m i:4 m A:3 W IH1 L ts6 j ax n4 ts6_h y4"], "retry": 2}']
    sys.argv += ['--char_to_similarity_bert_ids_file_path', '/home/hsiehpinghan/git/esun_ai_2022_summer_demo/model-spec/data/char_to_similarity_bert_ids.json']    
    sys.argv += ['--checkpoints', '/home/hsiehpinghan/git/esun_ai_2022_summer/model/sub_model_0',
                                  '/home/hsiehpinghan/git/esun_ai_2022_summer/model/sub_model_1',
                                  '/home/hsiehpinghan/git/esun_ai_2022_summer/model/sub_model_2',
                                  '/home/hsiehpinghan/git/esun_ai_2022_summer/model/sub_model_3',
                                  '/home/hsiehpinghan/git/esun_ai_2022_summer/model/sub_model_4']    
    sys.argv += ['--device', 'cpu']    
    """
    main(args=parse_args())