import os
import sys
import json
import torch
import hashlib

from absl import logging
from flask import Blueprint, jsonify, request
from datetime import datetime
from inference.inference import get_answer, get_tokenizer, get_model

bp = Blueprint('api', __name__, url_prefix='')
checkpoints = ['/home/hsiehpinghan/git/esun_ai_2022_summer/model/sub_model_0',
               '/home/hsiehpinghan/git/esun_ai_2022_summer/model/sub_model_1',
               '/home/hsiehpinghan/git/esun_ai_2022_summer/model/sub_model_2',
               '/home/hsiehpinghan/git/esun_ai_2022_summer/model/sub_model_3',
               '/home/hsiehpinghan/git/esun_ai_2022_summer/model/sub_model_4'] 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = get_tokenizer(checkpoints=checkpoints)
model = get_model(checkpoints=checkpoints,
                  device=device)

@bp.route('/inference', methods=['POST'])
def inference():
    data = request.get_json(force=True)
    answer = get_answer(tokenizer=tokenizer,
                        model=model,
                        request=data,
                        char_to_similarity_bert_ids_file_path='/home/hsiehpinghan/git/esun_ai_2022_summer_demo/model-spec/data/char_to_similarity_bert_ids.json',
                        device=device)
    resp = {'esun_uuid': data['esun_uuid'],
            'server_uuid': generate_server_uuid(),
            'server_timestamp': generate_server_timestamp(),
            'answer': answer}
    output = jsonify(resp)
    return output

def generate_server_uuid():
    s = hashlib.sha256()
    data = str(int(datetime.now().utcnow().timestamp())).encode('utf-8')
    s.update(data)
    server_uuid = s.hexdigest()
    return server_uuid

def generate_server_timestamp():
    return int(datetime.now().utcnow().timestamp())