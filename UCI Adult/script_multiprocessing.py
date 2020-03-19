import os
import itertools
import multiprocessing
import pandas as pd

df = pd.read_csv("../params.csv")

all_params = df["params"].tolist()[:25]

# d1_vals = [0.1, 0.2, 0.3, 0.4, 0.5]#, 0.6, 0.7, 0.8, 0.9, 1.0, 2, 3, 4, 5]
# d2_vals = [0.1, 0.2, 0.3, 0.4, 0.5]#, 0.6, 0.7, 0.8, 0.9, 1.0]
# epoch_vals = [500]

# # countinous_bound_vals = [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.5, 3, 4, 5, 10, 15, 20, 25, 50, 100, 150, 200, 250]
# # lambda__vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# countinous_bound_vals = [1]
# lambda__vals =  [0.5]





# d1_vals = [str(i) for i in d1_vals]
# d2_vals = [str(i) for i in d2_vals]
# countinous_bound_vals = [str(i) for i in countinous_bound_vals]
# lambda__vals = [str(i) for i in lambda__vals]
# epoch_vals = [str(i) for i in epoch_vals]

# a = itertools.product(d1_vals, d2_vals, countinous_bound_vals, lambda__vals, epoch_vals)
# all_params = []
# for i in a:
# 	all_params.append(list(i))
# all_params = [" ".join(i) for i in all_params]


def worker(param):
	"""thread worker function"""
	# print("\n\n###############################################################################\n\n")
	print("\n\nRUNNING CODE FOR ", param, "\n\n")
	# print("\n\n###############################################################################\n\n")
	os.system("python3.6 code.py " + param)
	# print("\n\n###############################################################################\n\n")
	print("\n\nCOMPLETED RUNNING FOR ", param, "\n\n")
	# print("\n\n###############################################################################\n\n")
	return

# if __name__ == '__main__':
# 	jobs = []
# 	for i in all_params:
# 		p = multiprocessing.Process(target=worker, args=(i,))
# 		jobs.append(p)
# 		p.start()


if __name__ == '__main__':
    pool = multiprocessing.Pool() #use all available cores, otherwise specify the number you want as an argument
    for i in all_params:
        pool.apply_async(worker, args=(i,))
    pool.close()
    pool.join()
