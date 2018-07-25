# -*- coding: utf-8 -*-
from pdf2text import pdf2text
import argparse
from pyltp import SentenceSplitter
import os
from pyltp import Segmentor
import re
import sys
from gensim.models import Word2Vec
import json
from multiprocessing import Pool

# 迭代器类
class DirofCorpus(object):
	def __init__(self, dirname):
		self.dirname = dirname

	def __iter__(self):
		with open(self.dirname) as f:
			line = f.readline()
			while line:
				single_text = json.loads(line)
				line = f.readline()
				for i in range(len(single_text)):
					yield single_text[i]

# 除去一些不是utf-8的乱码
def remove_invalid_utf8(data):
    new_data,count = re.subn('[x00-x08x10x0Bx0Cx0E-x19x7F]'
        + '|[x00-x7F][x80-xBF]+'
        + '|([xC0xC1]|[xF0-xFF])[x80-xBF]*'
        + '|[xC2-xDF]((?![x80-xBF])|[x80-xBF]{2,})'
        + '|[xE0-xEF](([x80-xBF](?![x80-xBF]))|(?![x80-xBF]{2})|[x80-xBF]{3,})', '!', data)

    new_data,count = re.subn('xE0[x80-x9F][x80-xBF]'
            + '|xED[xA0-xBF][x80-xBF]', '?', new_data)

    return new_data

# 简单的预处理，删除一些无意义的字符
def skip_some_char(sentences):
	skip_list = [' ', '\n', '=*', '.......']
	for skip_char in skip_list:
		sentences = sentences.replace(skip_char, "")
	return sentences

# 分句和分词
def ltp_sep(sentences, segmentor):
	tmp_sentences = SentenceSplitter.split(sentences)
	sentences_list = list(tmp_sentences)

	seg_list = []
	for i in range(len(sentences_list)):
		words = segmentor.segment(sentences_list[i])
		words_list = list(words)
		seg_list.append(words_list)

	return seg_list

# word2vec训练
def word2vec_train(text_path, words_dim, min_count, workers):
	text_path_list = os.listdir(text_path)
	model = Word2Vec(size=words_dim, min_count=min_count, workers=workers)
	for i in range(len(text_path_list)):
		all_text = DirofCorpus(text_path + text_path_list[i])
		if i == (len(text_path_list)-1):
			model.build_vocab(all_text, update=True, keep_raw_vocab=False)
		elif i == 0:
			model.build_vocab(all_text, update=False, keep_raw_vocab=True)
		else:
			model.build_vocab(all_text, update=True, keep_raw_vocab=True)

	for i in range(len(text_path_list)):
		all_text = DirofCorpus(text_path + text_path_list[i])
		model.train(all_text, total_examples=model.corpus_count, epochs=model.epochs)
	return model

# 从pdf提取文本，处理后保存
def pdf_to_text(k, data_path, text_save_path, wokers, cws_model_path):
	data_dirs = os.listdir(data_path)
	n = int(len(data_dirs) / wokers)
	
	segmentor = Segmentor()
	segmentor.load(cws_model_path)
	data_dirs = data_dirs[k*n:k*n+n]
	valid_pdf = 0

	for i in range(len(data_dirs)):
		path = data_path + data_dirs[i]
		try:
			tmp_str = pdf2text(path)
			tmp_str = skip_some_char(tmp_str)
			tmp_str = remove_invalid_utf8(tmp_str)
			sep_list = ltp_sep(tmp_str, segmentor)
			with open(text_save_path + 't' + str(k) + '.txt', 'a+') as save_f:
				json.dump(sep_list, save_f)
				save_f.write('\n')
		except:
			valid_pdf += 1

	print('----------pdf2text', i, 'finished, valid_pdf:', valid_pdf, '-----------------')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_path', 			default='./data/', 		help='path to load data', type=str)
	parser.add_argument('--words_dim', 			default=200 , 			help='dimension of word vectors', type=int)
	parser.add_argument('--text_save_path', 	default='./tmp_text/' , 	help='the path to save pdf_to_text', type=str)
	parser.add_argument('--is_first_time', 		default=False,			help='first time or not', type=bool)
	parser.add_argument('--min_count', 			default=1, 				help='min count',         type=int)
	parser.add_argument('--num_process', 		default=2, 				help='cpu nums to use', type=int)
	parser.add_argument('--model_save_path', 	default='./', 			help='the path to save model', type=str)
	parser.add_argument('--ltp_path', 			default='D:/BaiduNetdiskDownload/ltp-data-v3.3.1/ltp_data', help='ltp model path', type=str)

	args = parser.parse_args()

	cws_model_path = os.path.join(args.ltp_path, 'cws.model')
	data_path = args.data_path
	text_save_path = args.text_save_path
	num_process = args.num_process
	words_dim = args.words_dim
	min_count = args.min_count
	model_save_path = args.model_save_path

	print(cws_model_path)
	
	if args.is_first_time:
		p = Pool(num_process)
		for k in range(num_process):
			p.apply_async(pdf_to_text, args=(k, data_path, text_save_path, num_process, cws_model_path))
		p.close()
		p.join()

	model = word2vec_train(text_save_path, words_dim, min_count, num_process)
	model.save(model_save_path + "word2vec.model")

	# 读取模型和测试
	# model = Word2Vec.load(model_save_path + "word2vec.model")
	# print(model.most_similar('证券'))








	
