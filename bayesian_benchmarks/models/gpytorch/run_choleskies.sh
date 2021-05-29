
runSweep () {
	for split in 1 2 3 4;
	do
		#for cg in 0.001 0.01 0.1 1.0;
		#do
		#	for precond in 15 50 100;
		#	do
		#		echo $1 $2 $3 $split $cg $precond
		#		python runner.py --cg_tolerance=$cg --precond_size=$precond --dataset=$1 --split=$split --device=$2 \
		#			--database_path=results/ --with_adam --dtype=$3 >> tmp_keops
		#	done
		#done
		#echo $1 $2 $3 $split cholesky

		python runner.py --split=$split --device=$2 --is_cholesky --dataset=$1 --database_path=results/ --with_adam --dtype=$3
	done
}

runSweep wilson_bike 0 float &
runSweep wilson_elevators 2 float &
runSweep wilson_pol 4 float &
runSweep wilson_protein 6 float &
#gpu=0
#for dataset in wilson_bike wilson_elevators wilson_pol wilson_protein;
#do
#	echo $dataset $gpu
#	runSweep $dataset $gpu float &
#	gpu=$((gpu+1))
#	echo $dataset $gpu
#	runSweep $dataset $gpu double &
#	gpu=$((gpu+1))
#done
