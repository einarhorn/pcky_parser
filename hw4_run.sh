#!/bin/sh
treebank_filename=$1                # parses.train            : The name of the file holding the parsed sentences, one parse per line, in Chomsky Normal Form.
output_PCFG_file=$2                    # hw4_trained.pcfg        : The name of the file where the induced grammar should be written.
test_sentence_filename=$3            # sentences.txt            : The name of the file holding the test sentences to be parsed.
baseline_parse_output_filename=$4    # parses_base.out        : Output parses from your baseline PCFG parser
input_PCFG_file=$5                    # hw4_trained.pcfg      : The name of the file holding the induced PCFG grammar to be read.
improved_parse_output_filename=$6    # parses_improved.out    : Output parses from your improved PCFG parser
baseline_eval=$7                    # parses_base.eval        : evalb output for your baseline parses
improved_eval=$8                    # parses_improved.eval    : evalb output for your improved parses.
gold_standard_parse_file="/dropbox/18-19/571/hw4/data/parses.gold"

# Run grammar induction
./hw4_topcfg.sh $treebank_filename $output_PCFG_file

# Run pcfg
./hw4_parser.sh $output_PCFG_file $test_sentence_filename $baseline_parse_output_filename

# Evaluate on the base version
/dropbox/18-19/571/hw4/tools/evalb -p /dropbox/18-19/571/hw4/tools/COLLINS.prm $gold_standard_parse_file $baseline_parse_output_filename > $baseline_eval

# Run improved grammar induction
./hw4_improved_induction.sh $treebank_filename $input_PCFG_file

# Run improved pcfg
./hw4_improved_parser.sh $input_PCFG_file $test_sentence_filename $improved_parse_output_filename

# Evaluate inproved version
/dropbox/18-19/571/hw4/tools/evalb -p /dropbox/18-19/571/hw4/tools/COLLINS.prm $gold_standard_parse_file $baseline_parse_output_filename > $improved_eval
