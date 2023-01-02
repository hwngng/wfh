from collections import defaultdict

class Feature:
	class Application:
		KEY = 'application'
		def standardize(d):
			d = [x.lower() for x in d]
			return d
		def preprocess(features, **kwargs):
			taken_features = []
			total_words = sum(features.values())
			doc_freq = kwargs.get('doc_freq', defaultdict(int))
			total_docs = len(doc_freq)
			for feature, freq in features.items():
				if freq / total_words > 0.5 or (total_docs > 0 and doc_freq[feature] / total_docs > 0.9):
					continue
				taken_features.append(feature)
			return taken_features
	class FileEvent:
		KEY = 'file'
		def standardize(d):
			d = [x.lower() for x in d]
			return d
		def preprocess(features, **kwargs):
			taken_features = []
			total_words = sum(features.values())
			doc_freq = kwargs.get('doc_freq', defaultdict(int))
			total_docs = len(doc_freq)
			for feature, freq in features.items():
				if freq / total_words > 0.5 or (total_docs > 0 and doc_freq[feature] / total_docs > 0.9):
					continue
				taken_features.append(feature)
			return taken_features
	class Internet:
		KEY = 'internet'
		def standardize(d):
			d = [x.lower() for x in d]
			return d
		def preprocess(features, **kwargs):
			taken_features = []
			total_words = sum(features.values())
			doc_freq = kwargs.get('doc_freq', defaultdict(int))
			total_docs = len(doc_freq)
			for feature, freq in features.items():
				if freq / total_words > 0.5 or (total_docs > 0 and doc_freq[feature] / total_docs > 0.9):
					continue
				taken_features.append(feature)
			return taken_features
	ALL = [Application, FileEvent, Internet]
	COMBINATIONS = [[Application, Internet], [FileEvent]]
	KEY_MAP = {Application.KEY: Application, FileEvent.KEY: FileEvent, Internet.KEY: Internet}