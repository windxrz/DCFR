read -p "dataset (adult / compas): " dataset
read -p "task (DP / EO / CF): " task
coeff=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4 2.6 2.8 3.0 3.5 4.0 4.5 5.0 6.0 7.0 8.0 9.0 10.0 12.0 14.0 16.0 18.0 20.0)

for seed in {0..4}
do
    for fair in "${coeff[@]}"
    do
        python main.py --model LAFTR --fair-coeff $fair --seed $seed --dataset $dataset --task $task
    done
done
