#!/usr/bin/env python3
import sys
from nltk import load, word_tokenize, Tree, grammar
import numpy as np
import operator
import math



__authors__ = ['einarh', 'avijitv']

class ProbTreeNode():
    def __init__(self, node, log_probability):
        """
        :param node: a nonterminal node in the parse tree we create
        :type node: nltk.Nonterminal
        :param log_probability: real-value log probability of the subtree starting at this node
        :type node: float
        """
        self.node = node
        self.log_probability = log_probability
    
    def __eq__(self, other):
        if isinstance(other, grammar.Nonterminal):
            return self.node == other
        return self.node == other.node
    
    def __lt__(self, other):
        return self.log_probability < other.log_probability
    
    def __str__(self):
        return str(self.node).rstrip()
    
    def __repr__(self):
        return str(self.node).rstrip()
    

    

class PCKYParser:
    def __init__(self, grammar, beam_size):
        """
        :param grammar: PCFG in Chomsky-Normal-Form
        :type grammar: nltk.PCFG
        """
        self.grammar = grammar
        self.beam_size = beam_size

    def parse_sentence(self, sentence):
        """ Parse a raw sentence, and return a list of valid parses
        :type sentence: str
        :return: valid_parses
        :rtype: list(nltk.Tree)
        """
        tokenized_sentence = word_tokenize(text=sentence)
        return self.parse_tokenized_sentence(tokenized_sentence=tokenized_sentence)

    def parse_tokenized_sentence(self, tokenized_sentence):
        """ Parse a tokenized sentence, and return a list of valid parses
        :type tokenized_sentence: list(str)
        :return: valid_parses
        :rtype: list(nltk.Tree)
        """

        # Build our square PCKY table
        # Table will be (n + 1) x (n + 1), where n is len(tokenized_sentence)
        width = height = len(tokenized_sentence) + 1

        # Each cell [i][j] contains a list of subtrees
        table = [[list() for x in range(width)] for y in range(height)]
    
        # We fill from left to right, bottom to top, for the top right triangle
        for j in range(1, width):
            # First fill in a terminal cell
            current_word = tokenized_sentence[j - 1]
            table[j - 1][j] = self.__calculate_diagonal_cell(current_word=current_word)
            # Iterate over cells in upward direction from bottom-most cell
            for i in range(j - 1, -1, -1):
                for k in range(i + 1, j):
                    # Get a list of subtrees that are valid for the current span, 
                    # sorted by decreasing log probability
                    potential_subtrees = self.__calculate_intermediate_cell(table=table, i=i, j=j, k=k)

                    # Prune the number of subtrees in i,j to the top <beam_size> entries

                    if len(potential_subtrees) > 0:
                        table[i][j].extend(potential_subtrees)
                        table[i][j].sort(key=lambda x: x.label(), reverse=True)

                        # If using beam search, prune the result size
                        if self.beam_size != -1:
                            table[i][j] = table[i][j][:self.beam_size]
                            

        # Get the list of successful parses
        top_level_entries = table[0][width-1]
        valid_parses = [entry for entry in top_level_entries
                        if entry.label() == self.grammar.start()]

        # Re-sort the top level parses
        valid_parses.sort(key=lambda x: x.label(), reverse=True)
        
        # Return the highest log probability parse tree
        if len(valid_parses) > 0:
            # Strip parent annotations
            return self.__strip_parent_annotation(valid_parses[0])
        else:
            return None

    def __calculate_diagonal_cell(self, current_word):
        """ Return list of nltk.Tree objects, which correspond to nonterminals with current_word as a rhs terminal
        :type current_word: str
        :return: cell_contents
        :rtype: list(nltk.Tree)
        """

        # Search productions to find all lhs nonterminals that have current_word as a rhs terminal
        relevant_productions = self.grammar.productions(rhs=current_word)

        # Pretend this terminal is actually 'UNK' if there is no terminal matching the word
        if relevant_productions == []:
            relevant_productions = self.grammar.productions(rhs='UNK')

        # Create head nodes (nltk nonterminal + log probability) for each of the trees we will return
        tree_nodes = [ProbTreeNode(production.lhs(), self.__get_log_probability(production.prob())) 
                       for production in relevant_productions]

        # Sort tree nodes by decreasing log probability
        tree_nodes_sorted = sorted(tree_nodes, key=operator.attrgetter('log_probability'), reverse=True)

        # Convert to a list of nltk.Tree objects
        cell_contents = [Tree(node=tree_node, children=[current_word])
                         for tree_node in tree_nodes_sorted]
        return cell_contents

    def __calculate_intermediate_cell(self, table, i, j, k):
        """ Calculate intermediate cells, based on two previously calculated cells {A | A → BC ∈ grammar,
                                                                                        B ∈ table[ i, k ],
                                                                                        C ∈ table[ k, j ]}
        :param table: DP table storing intermediate subtrees
        :type table: list(list(nltk.Tree))
        :param i: Start of Left subtree span
        :type i: int
        :param j: End of Right subtree span
        :type j: int
        :param k: End of Left subtree span and Right subtree span
        :type k: int
        :return: cell_contents
        :rtype: list(nltk.Tree)
        """

        # List containing the subtrees to be returned
        cell_contents = list()

        # Get contents of the two cells we will be using
        first_tree_list = table[i][k]
        second_tree_list = table[k][j]

        # Calculate every combination of first and second cell
        for first_tree in first_tree_list:
            for second_tree in second_tree_list:
                # Get reference to the actual nltk nonterminals
                first_node = self.__get_nltk_nonterminal_from_tree(first_tree)
                second_node = self.__get_nltk_nonterminal_from_tree(second_tree)

                # Get probabilities of each of the trees
                first_log_probability = self.__get_log_probability_from_tree(first_tree)
                second_log_probability = self.__get_log_probability_from_tree(second_tree)

                # Check whether there is a production A -> B C, where B is first_list_item
                # and C is second_list_item
                relevant_productions = self.__get_productions_with_rhs(first_node, second_node)

                # Get LHS of each of the productions as node
                tree_nodes = [ProbTreeNode(production.lhs(), 
                                           self.__get_log_probability(production.prob()) + first_log_probability + second_log_probability) 
                              for production in relevant_productions]

                # Convert each LHS entry to a valid subtree
                subtrees = [Tree(node=tree_node,
                                 children=[first_tree, second_tree])
                            for tree_node in tree_nodes]

                # Add to list of valid subtrees created so far
                cell_contents += subtrees

        # Sort cell_contents by their log probabilities
        cell_contents.sort(key=lambda x: x.label(), reverse=True)

        return cell_contents

    def __get_productions_with_rhs(self, first_rhs, second_rhs):
        """ Get all productions in the grammar that have both of the given RHS items
            This method is used, since NLTK does not support searching for multiple RHS items
        :type first_rhs: nltk.Nonterminal
        :type second_rhs: nltk.Nonterminal
        :rtype: list(nltk.Production)
        """
        # Get all productions that contain the first_rhs nonterminal on its RHS
        productions_with_first_rhs = self.grammar.productions(rhs=first_rhs)

        # Get all productions of the form -> first_rhs second_rhs
        relevant_productions = []
        for candidate_prod in productions_with_first_rhs:
            if len(candidate_prod.rhs()) == 2 and candidate_prod.rhs()[1] == second_rhs:
                relevant_productions.append(candidate_prod)
        return relevant_productions
    
    def __get_nltk_nonterminal_from_tree(self, tree):
        """ Helper method to return the nltk nonterminal belonging to the head
            of the given tree
        :type tree: nltk.Tree
        :rtype: nltk.Nonterminal
        """
        return tree.label().node

    def __get_log_probability_from_tree(self, tree):
        """ Helper method to return the log probability for the given tree
        :type tree: nltk.Tree
        :rtype: float
        """
        return tree.label().log_probability
    
    def __get_log_probability(self, probability):
        return math.log(probability)
    
    def __strip_parent_annotation(self, tree_node):
        for subtree in tree_node.subtrees():
            if isinstance(subtree.label().node, grammar.Nonterminal):
                # Fix parent annotation
                node_symbol = subtree.label().node.symbol()
                if "_Parent_" in node_symbol:
                    idx = node_symbol.find("_Parent_")
                    updated_node = grammar.Nonterminal(node_symbol[:idx])
                    subtree.set_label(updated_node)
        return tree_node
    


def main(grammar_filename, sentence_filename, output_filename, beam_search_size):
    # Load CNF grammar
    grammar = load(grammar_filename, format='pcfg')

    # Generate parser based on grammar
    parser = PCKYParser(grammar=grammar, beam_size=beam_search_size)

    # Iterate over sentences in sentence_filename, produce parses and write to file with output_filename
    with open(sentence_filename, 'r') as infile:
        number_parses = []
        with open(output_filename, 'w') as outfile:
            for line in infile.readlines():
                # Strip any trailing whitespace from line (including newlines)
                line = line.rstrip()

                valid_parse = parser.parse_sentence(sentence=line)
                splitted = str(valid_parse).split()
                flat_tree = ' '.join(splitted)

                if valid_parse is None:
                    outfile.write('\n')
                else:
                    outfile.write(str(flat_tree) + '\n')


if __name__ == "__main__":
    # Get number of args (-1 to exclude the original file being counted as arg)
    num_args = len(sys.argv) - 1

    # Verify correct number of args passed
    if num_args == 4:
        grammar_filename = sys.argv[1]
        sentence_filename = sys.argv[2]
        output_filename = sys.argv[3]
        beam_search_size = int(sys.argv[4])
    else:
        print("Invalid number of arguments. Expected:", file=sys.stderr)
        print("hw4_parser.sh <grammar_filename> <test_sentence_filename> <output_filename> <beam_search_size>", file=sys.stderr)
        print("If no beam search required, set beam_search_size to -1", file=sys.stderr)
        sys.exit(-1)
    main(grammar_filename, sentence_filename, output_filename, beam_search_size)
