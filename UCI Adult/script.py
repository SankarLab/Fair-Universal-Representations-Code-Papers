import os
import itertools

d1_vals = [5]
d2_vals = [5]
countinous_bound_vals = [5, 10, 25, 50]
lambda__vals = [0.4]
epoch_vals = [5]

d1_vals = [str(i) for i in d1_vals]
d2_vals = [str(i) for i in d2_vals]
countinous_bound_vals = [str(i) for i in countinous_bound_vals]
lambda__vals = [str(i) for i in lambda__vals]
epoch_vals = [str(i) for i in epoch_vals]

a = itertools.product(d1_vals, d2_vals, countinous_bound_vals, lambda__vals, epoch_vals)
all_params = []
for i in a:
    all_params.append(list(i))
all_params = [" ".join(i) for i in all_params]

for param in all_params:

	print("\n\n###############################################################################\n\n")
	print("RUNNING CODE FOR ", param)
	print("\n\n###############################################################################\n\n")
	
	os.system("python3.6 code.py " + param)
	
	print("\n\n###############################################################################\n\n")
	print("COMPLETED RUNNING FOR ", param)
	print("\n\n###############################################################################\n\n")




# for d1 in d1_vals:
# 	for d2 in d2_vals:
# 		for countinous_bound in countinous_bound_vals:
# 			for lambda_ in lambda__vals:
# 				for epoch in epoch_vals:
# 					print("\n\n###############################################################################\n\n")
# 					print("RUNNING CODE FOR ", str(d1) + " " + str(d2) + " " + str(countinous_bound) + " " + str(lambda_) + " " + str(epoch))
# 					print("\n\n###############################################################################\n\n")
					
# 					os.system("python3.6 code.py " + str(d1) + " " + str(d2) + " " + str(countinous_bound) + " " + str(lambda_) + " " + str(epoch))
					
# 					print("\n\n###############################################################################\n\n")
# 					print("COMPLETED RUNNING FOR ", str(d1) + " " + str(d2) + " " + str(countinous_bound) + " " + str(lambda_) + " " + str(epoch))
# 					print("\n\n###############################################################################\n\n")
# os.system("python3.6 code.py 0.2 0.05 30 0.5 20")



