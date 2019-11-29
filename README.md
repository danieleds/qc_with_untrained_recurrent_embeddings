This repository contains the code for the following paper:

"[Question Classification with Untrained Recurrent Embeddings](http://for.unipi.it/daniele_di_sarli/files/2019/11/2019_DiSarli_QuestionClassif.pdf)",  
D. Di Sarli, C. Gallicchio, A. Micheli,  
AI*IA 2019 – Advances in Artificial Intelligence.

## Running

The folder structure is organized on two levels.
At the top level we have the dataset (QC), and inside that folder
we have the models applied to that dataset.

To launch an experiment, for example a leaky ESN on the QC task, run this command
from the root of the project:

    $ python3 QC/leaky_esn  # ...args...

## Reproducing results

To reproduce the results in the paper, just run the following commands:

    $ python3 QC/leaky_esn --searches 0 --final-trials 10
    $ python3 QC/leaky_esn_ensemble --searches 0 --final-trials 10
    $ python3 QC/leaky_esn_attn --searches 0 --final-trials 10
    $ python3 QC/mygru --searches 0 --final-trials 10

## Hardware

It is advisable to run the code on a GPU with at least 16GB of memory. Alternatively, the code
can easily be modified to accumulate the gradients of smaller minibatches in the training loop.
