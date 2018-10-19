from collections import defaultdict
from nltk import CFG, Nonterminal, Tree
import sys

__author__ = ['avijitv', 'einarh']


class ProbabilisticCFG:
    def __init__(self, corpus_file):
        with open(corpus_file, 'r') as infile:
            self._corpus = [j.strip() for j in infile.readlines() if j.strip()]
        self._start_symbol = None
        self._count_lhs = defaultdict(int)
        self._probabilities = defaultdict(dict)
        self._cfg = None

    def induce_pcfg(self):
        for line in self._corpus:
            parse_tree = Tree.fromstring(line)

            if self._start_symbol is None:
                self._start_symbol = Nonterminal(parse_tree.label())
            elif self._start_symbol != Nonterminal(parse_tree.label()):
                print('Inconsistent start symbols in parses', sys.stderr)
                exit(-1)

            parse_productions = parse_tree.productions()
            for prod in parse_productions:
                self._count_lhs[prod.lhs()] = self._count_lhs.get(prod.lhs(), 0) + 1
                try:
                    self._probabilities[prod.lhs()][prod] += 1
                except KeyError:
                    self._probabilities[prod.lhs()][prod] = 1

        self._cfg = CFG(start=self._start_symbol,
                        productions=[p for nt in self._probabilities for p in self._probabilities[nt]])
        if not self._cfg.is_chomsky_normal_form():
            print('Grammar not in Chomsky-Normal-Form', sys.stderr)
            exit(-1)

        for nt in self._probabilities:
            for p in self._probabilities[nt]:
                self._probabilities[nt][p] /= self._count_lhs[nt]

    def write_pcfg(self, output_filename):
        with open(output_filename, 'w') as outfile:
            outfile.write('%sstart %s\n' % ('%', str(self._start_symbol)))

            for lhs in self._probabilities:
                total_probability_lhs = sum(self._probabilities[lhs][p] for p in self._probabilities[lhs])
                outfile.write('# %s Productions\t\tTotal Probability: %.3f\n' % (lhs, total_probability_lhs))
                for prod in self._probabilities[lhs]:
                    outfile.write('%s [%f]\n' % (prod, self._probabilities[lhs][prod]))


def main():
    # Get number of args (-1 to exclude the original file being counted as arg)
    num_args = len(sys.argv) - 1

    # Required number of args for program to run
    required_args = 2

    # Verify correct number of args passed
    if num_args >= required_args:
        treebank_filename = sys.argv[1]
        output_pcfg_filename = sys.argv[2]
        pcfg_obj = ProbabilisticCFG(corpus_file=treebank_filename)
        pcfg_obj.induce_pcfg()
        pcfg_obj.write_pcfg(output_filename=output_pcfg_filename)
    else:
        print("Invalid number of arguments. Expected: %d\tGiven: %d" % (required_args, num_args),
              file=sys.stderr)
        print("hw4_topcfg.sh <treebank_filename> <output_pcfg_filename>", file=sys.stderr)
        exit(-1)


if __name__ == '__main__':
    main()
