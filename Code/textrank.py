import networkx as nx
import math
from rouge import FilesRouge
from rouge import Rouge
import build_graph as bg
import sys
import nltk
from nltk import *


def all_weight(node,graph):
	weight = 0.0
	for edge in graph.out_edges(node):
		weight += graph[edge[0]][edge[1]]['weight']
	return weight


def get_score_of_node(node, graph, alpha = 0.85):
	node_score = 0.0
	for edge in graph.in_edges(node):
		try:
			useful = (float) (graph.node[edge[0]]['rank']) / (float) (all_weight(edge[0],graph))
		except ZeroDivisionError:
			useful = 0.0
		node_score += graph[edge[0]][edge[1]]['weight'] * (float) (useful)
	score = (1-alpha) + alpha * node_score
	score = round(score,6)
	return score


def ranking_nodes(graph,alpha=0.85, max_iter=100, tol=1e-06):
	for i in range(max_iter):
		sum_error = 0.0
		print 'iter', i
		for node in graph.nodes():
			last_score = graph.node[node]['rank']
			new_score = get_score_of_node(node, graph, alpha)
			graph.node[node]['rank'] = new_score
			sum_error += abs(last_score - new_score)
		if sum_error < len(graph) * tol:
			break
	return graph
			

def text_rank(graph, alpha = 0.85, max_iter=100, tol=1e-06):
	graph_result = ranking_nodes(graph, alpha, max_iter, tol)
	return	graph_result


def main():
	if len(sys.argv) > 1:
		inputfile_name = sys.argv[1]
	else:
		print 'no file given'
	

	if len(sys.argv) >2:
		threshold = float(sys.argv[2])
	else:
		print 'no threshold value is given'

	if len(sys.argv) >3:
		golden_summary = sys.argv[3]
	else:
		 golden_summary = "output_file.txt"
	
	if len(sys.argv) >4:
		output_filename = sys.argv[4]
	else:
		output_filename = "output_file.txt"
	
	graph = bg.build_directed_graph(inputfile_name, threshold)    
	graph_result = text_rank(graph, alpha=0.85, max_iter=100, tol=1e-06)
	


	#print '----------------------------------------------- Summary  -----------------------------------------------'
	list_of_nodes = []
	for i  in range(1, len(graph_result)+1):
		list_of_nodes.append((graph_result.node[i]['rank'],graph_result.node[i]['node_id']))
	output_file = open(output_filename,"w")
	list_of_nodes = sorted(list_of_nodes, key=lambda x: -x[0])
	tokens = 0
	for i in range(len(list_of_nodes)):
			if tokens >= 250:
				break 
			tokens += len(nltk.word_tokenize(graph_result.node[list_of_nodes[i][1]]['value']))
			string = graph_result.node[list_of_nodes[i][1]]['value']	 
			output_file.write('%s' % string)
			#print str(i+1) + '.' + string 
	output_file.close()
	
	
	fp_hypo=open(output_filename,"r")
	fp_ref=open(golden_summary,"r")


	lines_hypo=fp_hypo.read()
	lines_ref=fp_ref.read()

	rouge=Rouge()
	scores = rouge.get_scores(lines_hypo, lines_ref, avg = True)
	print(scores)
	

if __name__ == '__main__':
    main()
