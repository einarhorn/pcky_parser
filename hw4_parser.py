#!/usr/bin/env python3
import sys
from nltk import load, word_tokenize, Tree, grammar
import numpy as np
import operator



__authors__ = ['einarh', 'avijitv']

class ProbTreeNode():
    def __init__(self, node, probability):
        self.node = node
        self.probability = probability
    
    def __eq__(self, other):
        if isinstance(other, grammar.Nonterminal):
            return self.node == other
        return self.node == other.node
    
    def __lt__(self, other):
        return self.probability < other.probability
    
    def __str__(self):
        return str(self.node)
    
    def __repr__(self):
        return str(self.node)
    

    

class PCKYParser:
    def __init__(self, grammar):
        """
        :param grammar: PCFG in Chomsky-Normal-Form
        :type grammar: nltk.PCFG
        """
        self.grammar = grammar

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
                    # sorted by decreasing probability
                    potential_subtrees = self.__calculate_intermediate_cell(table=table, i=i, j=j, k=k)

                    if len(potential_subtrees) > 0:
                        best_subtree = potential_subtrees[0]

                        # Store the most probable subtree in [i,j]
                        if len(table[i][j]) is 0:
                            table[i][j] = [best_subtree]
                        elif table[i][j][0].label() < best_subtree.label():
                            table[i][j] = [best_subtree]

        self.__visualize_table(table)

        # Return all top-level valid parses
        top_level_entries = table[0][width-1]
        valid_parses = [entry for entry in top_level_entries
                        if entry.label() == self.grammar.start()]
        return valid_parses

    def __calculate_diagonal_cell(self, current_word):
        """ Return list of nltk.Tree objects, which correspond to nonterminals with current_word as a rhs terminal
        :type current_word: str
        :return: cell_contents
        :rtype: list(nltk.Tree)
        """

        # Search productions to find all lhs nonterminals that have current_word as a rhs terminal
        relevant_productions = self.grammar.productions(rhs=current_word)

        # Create head nodes (nltk nonterminal + probability) for each of the trees we will return
        tree_nodes = [ProbTreeNode(production.lhs(), production.prob()) 
                       for production in relevant_productions]

        # Sort tree nodes by decreasing probability
        tree_nodes_sorted = sorted(tree_nodes, key=operator.attrgetter('probability'), reverse=True)

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
        first_list = table[i][k]
        second_list = table[k][j]

        # Calculate every combination of first and second cell
        for first_list_item in first_list:
            for second_list_item in second_list:
                # Check whether there is a production A -> B C, where B is first_list_item
                # and C is second_list_item
                relevant_productions = self.__get_productions_with_rhs(self.__get_nltk_nonterminal_from_tree(first_list_item),
                                                                       self.__get_nltk_nonterminal_from_tree(second_list_item))

                # Get LHS of each of the productions
                tree_nodes = [ProbTreeNode(production.lhs(), production.prob()) 
                              for production in relevant_productions]

                # Convert each LHS entry to a valid subtree
                subtrees = [Tree(node=tree_node,
                                 children=[first_list_item, second_list_item])
                            for tree_node in tree_nodes]

                # Add to list of valid subtrees created so far
                cell_contents += subtrees

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
        return tree.label().node
    
     # Pretty print the provided 2d list
    def __visualize_table(self, table):
        for row in table:
            row_contents = []
            for col in row:
                row_contents.append([cky_entry.label() for cky_entry in list(col)])
            print(row_contents)
                

def main(grammar_filename, sentence_filename, output_filename):
    # Load CNF grammar
    grammar = load(grammar_filename, format='pcfg')

    # Generate parser based on grammar
    parser = PCKYParser(grammar=grammar)

    # Iterate over sentences in sentence_filename, produce parses and write to file with output_filename
    with open(sentence_filename, 'r') as infile:
        number_parses = []
        with open(output_filename, 'w') as outfile:
            for line in infile.readlines():
                # Strip any trailing whitespace from line (including newlines)
                line = line.rstrip()
                print(line)
                outfile.write(line + '\n')
                valid_parses = parser.parse_sentence(sentence=line)
                for tree in valid_parses:
                    print(tree)
                    outfile.write(str(tree) + '\n')
                print('Number of parses: %d' % len(valid_parses))
                print()
                number_parses.append(len(valid_parses))
                outfile.write('Number of parses: %d\n\n' % len(valid_parses))
            avg_number_parses = np.mean(number_parses)
            print('Average number of parses: %.3f' % avg_number_parses)


if __name__ == "__main__":
    # Get number of args (-1 to exclude the original file being counted as arg)
    num_args = len(sys.argv) - 1

    # Required number of args for program to run
    required_args = 3

    # Verify correct number of args passed
    if num_args >= required_args:
        grammar_filename = sys.argv[1]
        sentence_filename = sys.argv[2]
        output_filename = sys.argv[3]
    else:
        print("Invalid number of arguments. Expected:", file=sys.stderr)
        print("hw3_parser.sh <grammar_filename> <test_sentence_filename> <output_filename>", file=sys.stderr)
        sys.exit(-1)
    main(grammar_filename, sentence_filename, output_filename)
