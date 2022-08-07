

for i in  3 5 7 9;
do
    # mvlens
    python LDS_tucker.py --epoch=150 --R_U=$i --num_fold=5 --method=Tucker-LDS --dataset=mvlens --machine=$USER --DAMPPING_U=0.90 --DAMPPING_gamma=0.3 --expand_odrer=two

    # dblp
    # python LDS_tucker.py --epoch=50 --R_U=$i --num_fold=5 --method=Tucker-LDS --dataset=dblp --machine=$USER --DAMPPING_U=0.5 --DAMPPING_gamma=0.3 --expand_odrer=two
done
