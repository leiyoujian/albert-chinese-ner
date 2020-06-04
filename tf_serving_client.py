import re
import os
import pickle
import tokenization
import numpy as np

def seg_char(sent):
    """
    把句子按字分开，不破坏英文结构
    """
    # 首先分割 英文 以及英文和标点
    pattern_char_1 = re.compile(r'([\W])')
    parts = pattern_char_1.split(sent)
    parts = [p for p in parts if len(p.strip())>0]
    # 分割中文
    pattern = re.compile(r'([\u4e00-\u9fa5])')
    chars = pattern.split(sent)
    chars = [w for w in chars if len(w.strip())>0]
    return chars

max_seq_length = 128
vocab_file = './albert_config/vocab.txt'
print(max_seq_length)
print(vocab_file)

tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)

with open(os.path.join('albert_base_ner_checkpoints', 'label2id.pkl'), 'rb') as rf:  # TODO need use label2id.pkl in model.tar.gz
    label2id = pickle.load(rf)
    id2label = {value: key for key, value in label2id.items()}
print(label2id)
print(id2label)

text = '因有关日寇在京掠夺文物详情，藏界较为重视，也是我们收藏北京史料中的要件之一。'
tokens = ['[CLS]']
tokens.extend(seg_char(text)[:max_seq_length-2])
tokens.append('[SEP]')
print(len(tokens), tokens)

input_ids = tokenizer.convert_tokens_to_ids(tokens)
for i in range(max_seq_length-len(tokens)):
    input_ids.append(0)
input_mask = [1 if i<len(tokens) else 0 for i in range(max_seq_length)]
segment_ids = [0 for _ in range(max_seq_length)]
label_ids = [0 for _ in range(max_seq_length)]
# input_ids = np.reshape(np.array(input_ids), (1, max_seq_length)).tolist()
# input_mask = np.reshape(np.array(input_mask), (1, max_seq_length)).tolist()
# segment_ids = np.reshape(np.array(segment_ids), (1, max_seq_length)).tolist()
# label_ids = np.reshape(np.array(label_ids), (1, max_seq_length)).tolist()
print(input_ids)
print(input_mask)


# pip install --upgrade tensorflow-serving-client==0.0.8

import sys
from tensorflow_serving_client.protos import predict_pb2, prediction_service_pb2
from grpc.beta import implementations
import tensorflow as tf
from tensorflow.python.framework import dtypes
import time

if __name__ == '__main__':
    start_time = time.time()
    channel = implementations.insecure_channel("localhost", 8501)
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = "albert_chinese_ner_model"
    request.inputs["input_ids"].ParseFromString(tf.contrib.util.make_tensor_proto(input_ids, dtype=dtypes.int32, shape=[1, max_seq_length]).SerializeToString())
    request.inputs["input_mask"].ParseFromString(tf.contrib.util.make_tensor_proto(input_mask, dtype=dtypes.int32, shape=[1, max_seq_length]).SerializeToString())
    request.inputs["segment_ids"].ParseFromString(tf.contrib.util.make_tensor_proto(segment_ids, dtype=dtypes.int32, shape=[1, max_seq_length]).SerializeToString())
    request.inputs["label_ids"].ParseFromString(tf.contrib.util.make_tensor_proto(label_ids, dtype=dtypes.int32, shape=[1, max_seq_length]).SerializeToString())
    response = stub.Predict(request, 10.0)  # TODO: BUG grpc.framework.interfaces.face.face.AbortionError: AbortionError(code=StatusCode.UNAVAILABLE, details="Trying to connect an http1.x server")
    results = {}
    for key in response.outputs:
        tensor_proto = response.outputs[key]
        nd_array = tf.contrib.util.make_ndarray(tensor_proto)
        results[key] = nd_array
    print("cost %ss to predict: " % (time.time() - start_time))
    print(results["pro"])
    print(results["classify"])
