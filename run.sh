# prepare todlist
# todlist.py s19,deep5 --dataset s19v1 > ../data/tods_s19_deep5.txt

# copy latest parameters as a template
# for ds in pa4_f150 pa4_f220 pa5_f090 pa5_f150 pa6_f090 pa6_f150; do
#     cp -v /projects/ACT/yilung/cuts/${ds}_s19_c11/cutparams_v1.par params/cutparams_${ds}.par
#     cp -v /projects/ACT/yilung/cuts/${ds}_s19_c11/cutParams_v1.par params/cutParams_${ds}.par
# done

# debug run
# python get_cuts.py --todlist ../data/tods_s19_deep5.txt --release release.yaml

# mpi run
# OMP_NUM_THREADS=4 mpirun -n 10 python get_cuts.py --todlist ../data/tods_s19_deep5.txt --release release.yaml --mpi

# report cuts
mkdir -p out
# python report_cuts.py --todlist ../data/tods_s19_deep5.txt --release release.yaml --oname report.txt
mpirun -n 10 python report_cuts.py --todlist ../data/tods_s19_deep5.txt --release release.yaml --oname report.txt --mpi
