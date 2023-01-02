from collections import Counter, defaultdict
import itertools
from feature import Feature


def calc_doc_frequency (docs):
	df = Counter()
	for doc in docs:
		for feature in set(doc.keys()):
			df[feature] += 1
	return df

def apply_mapping (arr, mapping, formatter=lambda x: x, **kwargs):
	formatter_args = kwargs.get('formatter_args', {})
	return [formatter(mapping[k], formatter_args) for k in arr]

def key_formatter (s, kawrgs):
	key = kawrgs.get('key', None)
	if key:
		return f'@{key}_{s}'
	else:
		return s

def create_combs (dikt_of_arr, keys, vocab=defaultdict(int)):
	comb_str = []
	n = len(keys)
	if n == 0:
		return []
	if n == 1:
		return apply_mapping(dikt_of_arr[keys[0]], vocab[keys[0]], formatter=key_formatter, formatter_args={'key': keys[0]})
	i = 1
	ret = apply_mapping(dikt_of_arr[keys[0]], vocab[keys[0]], formatter=key_formatter, formatter_args={'key': keys[0]})
	for i in range(1, n):
		ret = itertools.product(ret, apply_mapping(dikt_of_arr[keys[i]], vocab[keys[i]], formatter=key_formatter, formatter_args={'key': keys[i]}))
		ret = [' '.join(x) for x in ret]
	
	return ret

def preprocess (train_set):
	doc_freq = {}
	for metadata in Feature.ALL:
		train_set_by_cat = [x[metadata.KEY] for x in train_set]
		doc_freq[metadata.KEY] = calc_doc_frequency(train_set_by_cat)
	for data_point in train_set:
		for type_key, features in data_point.items():
			data_point[type_key] = Feature.KEY_MAP[type_key].preprocess(features, doc_freq=doc_freq[type_key])
	return train_set

def self_return (x):
	return x
	