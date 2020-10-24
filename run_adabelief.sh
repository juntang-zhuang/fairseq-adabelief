#! /bin/bash
SAVE=adabelief
AVERAGE=5
mkdir -p $SAVE

CUDA_VISIBLE_DEVICES=1 fairseq-train  \
          data-bin/iwslt14.tokenized.de-en   \
          --arch transformer_iwslt_de_en --share-decoder-input-output-embed  \
          --optimizer adabelief --adam-betas '(0.9, 0.999)' --adam-eps 1e-16 \
          --weight-decouple True --rectify True  --clip-norm 0.0   \
          --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000   \
          --dropout 0.3 --weight-decay 0.0001  --criterion label_smoothed_cross_entropy \
          --label-smoothing 0.1  --max-tokens 4096     --eval-bleu  \
          --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
          --eval-bleu-detok moses     --eval-bleu-remove-bpe     --eval-bleu-print-samples  \
          --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
          --save-dir $SAVE --max-epoch 50 \
          | tee -a $SAVE/log.txt

python scripts/average_checkpoints.py --inputs $SAVE --num-epoch-checkpoints $AVERAGE --output $SAVE/average_model.pt


CUDA_VISIBLE_DEVICES=1 fairseq-generate data-bin/iwslt14.tokenized.de-en \
    --path $SAVE/average_model.pt \
    --batch-size 128 --beam 5 --remove-bpe \
    | tee -a $SAVE/result.txt

