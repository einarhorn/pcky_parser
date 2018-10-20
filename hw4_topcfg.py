from collections import defaultdict
from nltk import Nonterminal, Production, Tree
import sys

__author__ = ['avijitv', 'einarh']


class PCFG:
    def __init__(self, corpus_file):
        with open(corpus_file, 'r') as infile:
            self._corpus = [j.strip() for j in infile.readlines() if j.strip()]
        self._train_corpus = self._corpus
        self.__start_symbol = None
        self.__count_lhs = defaultdict(int)
        self.__probabilities = defaultdict(dict)
        # self.__cfg = None

    @staticmethod
    def parse_productions(parse_tree):
        return parse_tree.productions()

    def induce_pcfg(self):
        for line in self._train_corpus:
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

        """
        self.__cfg = CFG(start=self.__start_symbol,
                         productions=[p for nt in self.__probabilities for p in self.__probabilities[nt]])
        if not self.__cfg.is_chomsky_normal_form():
            print('Grammar not in Chomsky-Normal-Form', sys.stderr)
            exit(-1)
        """

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


class ParentAnnotatedPCFG(PCFG):
    def __init__(self, treebank_file):
        super().__init__(treebank_file)

        self._train_corpus = self._corpus

    def parse_productions(self, parse_tree, parent_label='', parent_annotation_level='non-preterminal'):
        if not parse_tree:
            return []
        elif len(parse_tree) == 1:
            if parent_annotation_level == 'non-preterminal':
                updated_lhs = Nonterminal(parse_tree.label())
            elif parent_annotation_level == 'all':
                updated_lhs = Nonterminal(parse_tree.label() + '_Parent_' + parent_label)
            else:
                updated_lhs = Nonterminal(parse_tree.label())
            rhs = [parse_tree[0]]
            return [Production(lhs=updated_lhs, rhs=rhs)]

        productions = []
        updated_rhs = []
        for i in parse_tree:
            updated_rhs.append(Nonterminal(i.label() + '_Parent_' + parse_tree.label()))
            productions += self.parse_productions(parse_tree=i, parent_label=parse_tree.label())

        if not parent_label:
            parent_label = 'NULL'

        updated_lhs = Nonterminal(parse_tree.label() + '_Parent_' + parent_label)
        productions = [Production(lhs=updated_lhs, rhs=updated_rhs)] + productions
        return productions


class OOVHandledPCFG(ParentAnnotatedPCFG):
    def __init__(self, treebank_file, val_ratio=0.1):
        super().__init__(treebank_file=treebank_file)
        self._val_size = int(len(self._corpus) * val_ratio)
        self._train_corpus = self._corpus[:-1 * self._val_size]
        self._val_corpus = self._corpus[-1 * self._val_size:]

        self._val_vocab = self.get_vocab(parsed_sentences=self._val_corpus)
        self._train_vocab = self.get_vocab(parsed_sentences=self._train_corpus)
        self._oov_vocab = self._val_vocab.difference(self._train_vocab)
        print('Train vocab: %d\tVal vocab: %d\tOOV vocab: %d' % (len(self._train_vocab),
                                                                 len(self._val_vocab),
                                                                 len(self._oov_vocab)))
        self._substitute_oov()

    @staticmethod
    def get_vocab(parsed_sentences):
        vocab = set([])
        for parse_str in parsed_sentences:
            parse_tree = Tree.fromstring(parse_str)
            terminals = set(parse_tree.leaves())
            vocab = vocab.union(terminals)
        return vocab

    def _substitute_oov(self):
        added_corpus = []
        for parse_str in self._val_corpus:
            parse_tree = Tree.fromstring(parse_str)
            for pos in parse_tree.treepositions('leaves'):
                if parse_tree[pos] in self._oov_vocab:
                    parse_tree[pos] = 'UNK'
            added_corpus.append(str(parse_tree))
        self._train_corpus += added_corpus
        self._val_corpus.clear()


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
        elif mode == 'pa':
            improved_pcfg_obj = ParentAnnotatedPCFG(treebank_filename)
            improved_pcfg_obj.induce_pcfg()
            improved_pcfg_obj.write_pcfg(output_filename=output_pcfg_filename)
        elif mode == 'pa_all+oov':
            oov_pcfg_obj = OOVHandledPCFG(treebank_file=treebank_filename)
            oov_pcfg_obj.induce_pcfg()
            oov_pcfg_obj.write_pcfg(output_filename=output_pcfg_filename)
        else:
            print('Invalid mode argument.\tExpected one of naive or improved.\tGiven: %s' % mode)

    else:
        print("Invalid number of arguments. Expected: %d\tGiven: %d" % (required_args, num_args),
              file=sys.stderr)
        print("hw4_topcfg.py <treebank_filename> <output_pcfg_filename> <naive|improved>", file=sys.stderr)
        exit(-1)


if __name__ == '__main__':
    main()
