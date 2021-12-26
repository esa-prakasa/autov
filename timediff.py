import time
import os

os.system("cls")

# Blok pertama 
time_start = time.perf_counter()
for i in range(100):
    k = i**3 * i**3
    l = i**3 * i**3
    m = i**3 * i**3
    n = i**3 * i**3
    o = i**3 * i**3
    p = (k * l * m * n * o)**100

time_elapsed = (time.perf_counter() - time_start)
print ("Waktu Blok pertama: %5.5f secs " % (time_elapsed))

# Blok kedua 
time_start = time.perf_counter()
for i in range(10000):
    k = i**3 * i**3
    l = i**3 * i**3
    m = i**3 * i**3
    n = i**3 * i**3
    o = i**3 * i**3
    p = (k * l * m * n * o)**100

time_elapsed = (time.perf_counter() - time_start)
print ("Waktu Blok kedua: %5.5f secs " % (time_elapsed))
