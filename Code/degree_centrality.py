import networkx as nx
import math
from rouge import FilesRouge
from rouge import Rouge
import build_graph as bg
import sys
import nltk
from nltk import *


def degree_centrality(graph):
	for node in graph.nodes():
		graph.node[node]['rank'] = len(graph.edges(node))
	return	graph


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
		print 'no ground truth is given'
	
	if len(sys.argv) >4:
		output_filename = sys.argv[4]
	else:
		output_filename = "output_file.txt"
	

	graph = bg.build_undirected_graph(inputfile_name, threshold)    
	graph_result = degree_centrality(graph)


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
