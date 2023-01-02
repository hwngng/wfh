import json
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections import defaultdict, Counter
from feature import Feature
import itertools
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import common
import re
from sklearn.ensemble import RandomForestClassifier
import pickle

class BaseModel:
	def __init__(self):
		self._encoders = {}
		self._vocab = None
		self._list_features = None
		pass

	def _take_in_vocab (self, key_type, features):
		vocab = self._vocab[key_type]
		taken_features = []
		for feature in features:
			if feature in vocab:
				taken_features.append(feature)
		return taken_features
	
	def _transform_features (self, dataset, is_training = True):
		if is_training:
			self._encoders = {}
			self._vocab = {}
			self._list_features = {}
			for metadata in Feature.ALL:
				features_by_cat = [x[metadata.KEY] for x in dataset]
				encoder = CountVectorizer(preprocessor=common.self_return, tokenizer=common.self_return)
				encoder.fit(features_by_cat)
				self._encoders[metadata.KEY] = encoder
				self._vocab[metadata.KEY] = encoder.vocabulary_
				self._list_features[metadata.KEY] = encoder.get_feature_names_out()
		else:
			for data_point in dataset:
				for key_type, features in data_point.items():
					data_point[key_type] = self._take_in_vocab(key_type, features)
		with open('vocab.json', 'w') as f:
			f.writelines(str(self._list_features))
		transformed_features = []
		for data_point in dataset:
			features_comb = []
			for comb in Feature.COMBINATIONS:
				joined_features = common.create_combs(data_point, [metadata.KEY for metadata in comb], vocab=self._vocab)
				features_comb.extend(joined_features)
			transformed_features.append(features_comb)
		return transformed_features
	def _decode_feature (self, key_type, idx):
		return self._list_features[key_type][idx]

class DecisionTreeModel(BaseModel):
	reVecFeature = re.compile(r'feature_(\d+)')
	reEncFeature = re.compile(r'@(\w+)_(\d+)')

	def __init__ (self):
		self._encoder = CountVectorizer(preprocessor=common.self_return, tokenizer=common.self_return)
		self._model = DecisionTreeClassifier(splitter='best')
		pass

	def fit (self, train_set, label_set):
		transformed_train = self._transform_features(train_set)	
		vectorized_train_set = self._encoder.fit_transform(transformed_train)
		self._model.fit(vectorized_train_set, label_set)
		pass

	def predict (self, dataset):
		transformed_features = self._transform_features(dataset, is_training=False)
		vectorized_features = self._encoder.transform(transformed_features)
		results = self._model.predict(vectorized_features)
		return results

	def _regex_match_enc(self, matchedObj):
		key_type = matchedObj.group(1)
		idx = int(matchedObj.group(2))
		return self._decode_feature(key_type, idx)

	def _get_feature_name (self, idx):
		encoder_feature_names = self._encoder.get_feature_names_out()
		encoded_feature = encoder_feature_names[idx]
		orig_feature = self.reEncFeature.sub(repl=self._regex_match_enc, string=encoded_feature)
		return orig_feature
	
	def _regex_match_vec (self, matchedObj):
		feature_idx = int(matchedObj.group(1))
		feature_name = self._get_feature_name(feature_idx)
		return feature_name

	def explain (self):
		debug_txt = tree.export_text(self._model)
		feature_named_debug = self.reVecFeature.sub(repl=self._regex_match_vec, string=debug_txt)
		return feature_named_debug

class RandomForestModel(BaseModel):
	reVecFeature = re.compile(r'feature_(\d+)')
	reEncFeature = re.compile(r'@(\w+)_(\d+)')

	def __init__ (self):
		self._encoder = CountVectorizer(preprocessor=common.self_return, tokenizer=common.self_return)
		self._model = RandomForestClassifier(n_estimators=5, bootstrap=False)
		pass

	def fit (self, train_set, label_set):
		transformed_train = self._transform_features(train_set)	
		vectorized_train_set = self._encoder.fit_transform(transformed_train)
		self._model.fit(vectorized_train_set, label_set)
		pass

	def predict (self, dataset):
		transformed_features = self._transform_features(dataset, is_training=False)
		vectorized_features = self._encoder.transform(transformed_features)
		results = self._model.predict(vectorized_features)
		return results

	def _regex_match_enc(self, matchedObj):
		key_type = matchedObj.group(1)
		idx = int(matchedObj.group(2))
		return self._decode_feature(key_type, idx)

	def _get_feature_name (self, idx):
		encoder_feature_names = self._encoder.get_feature_names_out()
		encoded_feature = encoder_feature_names[idx]
		orig_feature = self.reEncFeature.sub(repl=self._regex_match_enc, string=encoded_feature)
		return orig_feature
	
	def _regex_match_vec (self, matchedObj):
		feature_idx = int(matchedObj.group(1))
		feature_name = self._get_feature_name(feature_idx)
		return feature_name

	def explain (self):
		explain_txt = ''
		for t in self._model.estimators_:
			debug_txt = tree.export_text(t)
			feature_named_debug = self.reVecFeature.sub(repl=self._regex_match_vec, string=debug_txt)
			explain_txt += feature_named_debug + '\n'
		return explain_txt



label_set = ["dev", "dev", "dev", "dev", "test", "ba"]
f = open('train.json', 'r')
train_set = json.load(f)
train_set_cnt = []

for data_point in train_set:
	data_point_cnt = {}
	for key, value in data_point.items():
		data_point_cnt[key] = dict(Counter(value))
	train_set_cnt.append(data_point_cnt)

train_set = common.preprocess(train_set_cnt)	

model = DecisionTreeModel()
model.fit(train_set, label_set)
print(model.explain())
pickle.dump(model, open('model.sav', 'wb'))
saved_model = pickle.load(open('model.sav', 'rb'))
print(model.predict(train_set))