# Test Transformer with AdaBelief Optimizer
This repo is based on ```fairseq``` repository (lates version) https://github.com/pytorch/fairseq <br>
An implementation of AdaBelief optimizer compatible with ```fairseq``` is in ```fairseq/optim```

## Dependencies
```PyTorch==1.6.0```

## How to run on IWSLT14 DE-EN
### Install current package
```pip install --editable .```

### Prepare data
```
cd examples/translation/
bash prepare-iwslt14.sh
cd ../..
```
### Preprocess/binarize the data
```
TEXT=examples/translation/iwslt14.tokenized.de-en
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.de-en \
    --workers 20
```
### run with Adam optimizer
```sh run_adam.sh```

### run with AdaBelief optimizer
```sh run_adabelief.sh```

## Results (BLEU score)
| Adam      | Adabelief |
| --------- | --------- |
| 35.02     | 35.17       |
