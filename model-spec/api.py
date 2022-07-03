import torch
import hashlib

from flask import request
from flask import jsonify
from flask import Blueprint
from datetime import datetime
from inference.inference import get_model
from inference.inference import get_answer
from inference.inference import get_tokenizer

bp = Blueprint('api', __name__, url_prefix='')
checkpoints = ['/content/checkpoint/0',
               '/content/checkpoint/1',
               '/content/checkpoint/2',
               '/content/checkpoint/3',
               '/content/checkpoint/4'] 
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
                        char_to_similarity_bert_ids_file_path='/content/data/char_to_similarity_bert_ids.json',
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