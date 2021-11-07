# sber-default

Sberbank default hackathon solution

<a href="https://dsbattle.com/hackathons/gsb/">
  <img alt="GSB" src="https://dsbattle.com/hackathons/gsb/assets/images/gsb-main.png" width=600" height="450">
</a>

## Install:

```shell
pip install -U -r requirements.txt
```

## Demo Local Up:

```shell
streamlit run demo/app_fin.py # for fin
streamlit run demo/app_no_fin.py # for no_fin
```

## Training:

```shell
python train.py --config=./configs/forest.yml \
                --work_dir=/home/sleep3r/sberruns \
                --validation.cutoff=0.04
```

## Grid training from Terminal:

```shell
declare -a penaltyes=("l1" "l2") 

for penalty in "${penaltyes[@]}"                                            
do
    python3 train.py --config=./configs/baseline.yml \
                     --model.params.penalty="$penalty"
done
```

```shell
for C in 1 2                                           
do
    python3 train.py --config=./configs/baseline.yml \
                     --model.params.C $C
done
```
