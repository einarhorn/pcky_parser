from collections import defaultdict
from nltk import CFG, Nonterminal, Production, Tree
import sys

__author__ = ['avijitv', 'einarh']


class PCFG:
    def __init__(self, corpus_file):
        with open(corpus_file, 'r') as infile:
            self._corpus = [j.strip() for j in infile.readlines() if j.strip()]
        self.__start_symbol = None
        self.__count_lhs = defaultdict(int)
        self.__probabilities = defaultdict(dict)
        self.__cfg = None

    @staticmethod
    def parse_productions(parse_tree):
        return parse_tree.productions()

    def induce_pcfg(self):
        for line in self._corpus:
            parse_tree = Tree.fromstring(line)
            productions = self.parse_productions(parse_tree)
            if self.__start_symbol is None:
                self.__start_symbol = productions[0].lhs()
            elif self.__start_symbol != productions[0].lhs():
                print('Inconsistent start symbols', sys.stderr)

            for prod in productions:
                self.__count_lhs[prod.lhs()] = self.__count_lhs.get(prod.lhs(), 0) + 1
                try:
                    self.__probabilities[prod.lhs()][prod] += 1
                except KeyError:
                    self.__probabilities[prod.lhs()][prod] = 1

        self.__cfg = CFG(start=self.__start_symbol,
                         productions=[p for nt in self.__probabilities for p in self.__probabilities[nt]])
        if not self.__cfg.is_chomsky_normal_form():
            print('Grammar not in Chomsky-Normal-Form', sys.stderr)
            exit(-1)

        for nt in self.__probabilities:
            for p in self.__probabilities[nt]:
                self.__probabilities[nt][p] /= self.__count_lhs[nt]

    def write_pcfg(self, output_filename):
        with open(output_filename, 'w') as outfile:
            outfile.write('%sstart %s\n' % ('%', str(self.__start_symbol)))

            for lhs in self.__probabilities:
                total_probability_lhs = sum(self.__probabilities[lhs][p] for p in self.__probabilities[lhs])
                outfile.write('# %s Productions\t\tTotal Probability: %.3f\n' % (lhs, total_probability_lhs))
                for prod in self.__probabilities[lhs]:
                    outfile.write('%s [%f]\n' % (prod, self.__probabilities[lhs][prod]))
                outfile.write('\n\n')


class ImprovedPCFG(PCFG):
    def __init__(self, treebank_file):
        super().__init__(treebank_file)

    def parse_productions(self, parse_tree, parent_label=''):
        if not parse_tree:
            return []
        elif len(parse_tree) == 1:
            updated_lhs = Nonterminal(parse_tree.label() + '_Parent:' + parent_label)
            rhs = [parse_tree[0]]
            return [Production(lhs=updated_lhs, rhs=rhs)]

        productions = []
        updated_rhs = []
        for i in parse_tree:
            updated_rhs.append(Nonterminal(i.label() + '_Parent:' + parse_tree.label()))
            productions += self.parse_productions(parse_tree=i, parent_label=parse_tree.label())

        if not parent_label:
            parent_label = 'NULL'

        updated_lhs = Nonterminal(parse_tree.label() + '_Parent:' + parent_label)
        productions = [Production(lhs=updated_lhs, rhs=updated_rhs)] + productions
        return productions


def main():
    # Get number of args (-1 to exclude the original file being counted as arg)
    num_args = len(sys.argv) - 1

    # Required number of args for program to run
    required_args = 3

    # Verify correct number of args passed
    if num_args >= required_args:
        treebank_filename = sys.argv[1]
        output_pcfg_filename = sys.argv[2]
        mode = sys.argv[3]

        if mode == 'naive':
            pcfg_obj = PCFG(corpus_file=treebank_filename)
            pcfg_obj.induce_pcfg()
            pcfg_obj.write_pcfg(output_filename=output_pcfg_filename)
        elif mode == 'improved':
            improved_pcfg_obj = ImprovedPCFG(treebank_filename)
            improved_pcfg_obj.induce_pcfg()
            improved_pcfg_obj.write_pcfg(output_filename=output_pcfg_filename)
        else:
            print('Invalid mode argument.\tExpected one of naive or improved.\tGiven: %s' % mode)
    else:
        print("Invalid number of arguments. Expected: %d\tGiven: %d" % (required_args, num_args),
              file=sys.stderr)
        print("hw4_topcfg.py <treebank_filename> <output_pcfg_filename> <naive|improved>", file=sys.stderr)
        exit(-1)


if __name__ == '__main__':
    main()
