# sber-default

Sberbank default hackathon solution

<a href="https://dsbattle.com/hackathons/gsb/">
  <img alt="GSB" src="https://dsbattle.com/hackathons/gsb/assets/images/gsb-main.png" width=600" height="450">
</a>

## Install:

```shell
pip install -U -r requirements.txt
```

## Training:

```shell
python3 train.py --config=./configs/baseline.yml
```

## Grid Training:

```shell
declare -a penaltyes=("l1" "l2") 

for penalty in "${penaltyes[@]}"                                            
do
    python3 train.py --config=./configs/baseline.yml \
                     --model.params.penalty="$penalty"
done
```