import time

# starting time
start = time.time()

# program body starts
#for i in range(10):
#    print(i)

# sleeping for 1 sec to get 10 sec runtime
time.sleep(90)

# program body ends

# end time
end = time.time()

# total time taken

durSec = end - start
durMin = durSec/60
print(f"Runtime of the program is {durSec} second")
print(f"Runtime of the program is {durMin} minute")