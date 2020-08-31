read -p "dataset (adult / compas): " dataset

for seed in {0..4}
do
    python ../main.py --model UNFAIR --seed $seed --dataset $dataset
done
