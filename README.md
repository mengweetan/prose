# Haiku emb
Usage:

```
pip install -r requirements.txt

- build embedding matrix; add -b b
python train/train.py -o ./model/haiku/ -b b
- at paperspace
python train/train.py -o /storage

python deploy/infer.py -m ./model/haiku/model.pkl

```
