# sber-default

Sberbank default hackathon solution

<div align="center">
  [![GSB](https://dsbattle.com/assets/images/dsbattle-logo4.png)](https://dsbattle.com/hackathons/gsb/)

  <a href="https://sber-default-demo.herokuapp.com"><h2>--> Interactive Demo <--</h2></a>
</div>
  
## Install:

```shell
pip install -U -r requirements.txt
```

## Demo Local Up:

```shell
streamlit run demo/app.py
```

## Training:

```shell
python train.py --config=./configs/lgbm_fin.yml \
                --work_dir=/home/sleep3r/sberruns 
                
python train.py --config=./configs/lgbm_no_fin.yml \
                --work_dir=/home/sleep3r/sberruns 
```

## Grid training from Terminal:

```shell
declare -a penaltyes=("l1" "l2") 

for penalty in "${penaltyes[@]}"                                            
do
    python3 train.py --config=./configs/logreg.yml \
                     --model.params.penalty="$penalty"
done
```

```shell
for C in 1 2                                           
do
    python3 train.py --config=./configs/logreg.yml \
                     --model.params.C $C
done
```

## Merge submits:
```shell
 python ./scripts/merge_submits.py --sub_path_1=/home/sleep3r/finsub.csv \
                                   --sub_path_2=/home/sleep3r/nofinsub.csv \
                                   --final_sub_path=/home/sleep3r/result_submit.csv

```
