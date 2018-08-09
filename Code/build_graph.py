from nltk import *
from nltk.corpus import stopwords
import networkx as nx
import math
from nltk.tokenize import sent_tokenize
import nltk
from string import punctuation
import sys


def strip_punctuation(s):
	return ''.join(c for c in s if c not in punctuation)

def make_lower(s):
	return ''.join(c.lower() for c in s)


stopwords = stopwords.words("english")
stemmer = SnowballStemmer("english")


def build_directed_graph(inputfile_name, threshold):

	input_file=open(inputfile_name, "r")
	sentences = []
	for line in input_file:
		if len(line) > 0:
			tokens = nltk.word_tokenize(line)
			if len(tokens) >= 20:
				sentences.append(line)
	input_file.close()
		
	total_sentences = len(sentences)
	vocab = {}
	original_sentences = sentences[:]
	processed_sentences = []
	vocab_index = {}
	
	i=0
	for sentence in sentences:
		sentence = strip_punctuation(sentence)
		sentence = make_lower(sentence)
		tokens = nltk.word_tokenize(sentence)
		filtered_tokens = [word for word in tokens if word not in (stopwords)]
		sentence = ' '.join(stemmer.stem(word) for word in filtered_tokens)
		unique_words = set(nltk.word_tokenize(sentence))
		processed_sentences.append(sentence)
		for word in unique_words:
			if word in vocab:
				vocab[word] += 1
			else:
				vocab[word] = 1
				vocab_index[word] = i
				i += 1


	graph = nx.DiGraph()
	node_id = 1
	tf_idf_vectors = [] 
	for i,sentence in enumerate(processed_sentences):
		tf_idf_vectors.append([ 0.0 for x in range(len(vocab))])
		words = nltk.word_tokenize(sentence)
		normalized_sum = 0.0
		for word in words:
			#tf = 1.0 + math.log10( (float) (words.count(word)))
			#tf = (float) (words.count(word))
			tf = (float) (words.count(word)) / len(words)
			idf = math.log10( (float)(total_sentences) / vocab[word])
			tf_idf_vectors[i][vocab_index[word]] = tf*idf
			normalized_sum += ((tf*idf) * (tf*idf))
		for j in range(len(tf_idf_vectors[i])):
			try:
				tf_idf_vectors[i][j] /= (float) (math.sqrt(normalized_sum))
			except ZeroDivisionError:
				tf_idf_vectors[i][j] /= (float) (1.0)
				
		graph.add_node(node_id,node_id=node_id,value = original_sentences[i],rank=1)	
		node_id =node_id+1


	for i in range(1,node_id):
		for j in range(i+1, node_id):
			sim_weight = 0.0
			for k in range(len(tf_idf_vectors[i-1])):
				sim_weight += (tf_idf_vectors[i-1][k] * tf_idf_vectors[j-1][k])
			if sim_weight > threshold / 10.0:
				graph.add_edge(i,j, weight = sim_weight) 
				graph.add_edge(j,i, weight = sim_weight)
	
	return graph


def build_undirected_graph(inputfile_name, threshold):

	input_file=open(inputfile_name, "r")
	sentences = []
	for line in input_file:
		if len(line) > 0:
			tokens = nltk.word_tokenize(line)
			if len(tokens) >= 20:
				sentences.append(line)
	input_file.close()
		
	total_sentences = len(sentences)
	vocab = {}
	original_sentences = sentences[:]
	processed_sentences = []
	vocab_index = {}
	
	i=0
	for sentence in sentences:
		sentence = strip_punctuation(sentence)
		sentence = make_lower(sentence)
		tokens = nltk.word_tokenize(sentence)
		filtered_tokens = [word for word in tokens if word not in (stopwords)]
		sentence = ' '.join(stemmer.stem(word) for word in filtered_tokens)
		unique_words = set(nltk.word_tokenize(sentence))
		processed_sentences.append(sentence)
		for word in unique_words:
			if word in vocab:
				vocab[word] += 1
			else:
				vocab[word] = 1
				vocab_index[word] = i
				i += 1


	graph = nx.Graph()
	node_id = 1
	tf_idf_vectors = [] 
	for i,sentence in enumerate(processed_sentences):
		tf_idf_vectors.append([ 0.0 for x in range(len(vocab))])
		words = nltk.word_tokenize(sentence)
		normalized_sum = 0.0
		for word in words:
			#tf = 1.0 + math.log10( (float) (words.count(word)))
			#tf = (float) (words.count(word))
			tf = (float) (words.count(word)) / len(words)
			idf = math.log10( (float)(total_sentences) / vocab[word])
			tf_idf_vectors[i][vocab_index[word]] = tf*idf
			normalized_sum += ((tf*idf) * (tf*idf))
		for j in range(len(tf_idf_vectors[i])):
			try:
				tf_idf_vectors[i][j] /= (float) (math.sqrt(normalized_sum))
			except ZeroDivisionError:
				tf_idf_vectors[i][j] /= (float) (1.0)
		graph.add_node(node_id,node_id=node_id,value = original_sentences[i],rank=1)	
		node_id =node_id+1


	for i in range(1,node_id):
		for j in range(i+1, node_id):
			sim_weight = 0.0
			for k in range(len(tf_idf_vectors[i-1])):
				sim_weight += (tf_idf_vectors[i-1][k] * tf_idf_vectors[j-1][k])
			if sim_weight > threshold / 10.0:
				graph.add_edge(i,j, weight = sim_weight)
	
	return graph
