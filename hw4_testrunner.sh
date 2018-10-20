#!/bin/sh
./hw4_run.sh data/parses.train hw4_trained.pcfg \
           data/sentences.txt parses_base.out \
           hw4_improved.pcfg \
           parses_improved.out \
		    parses_base.eval parses_improved.eval