import os

os.system("cls")

for i in range(10):
    try:
        x = 1/(5-i)
    except:
        x = 123456789
    print("%d  %3.2f"%(i,x))

for i in range(5):
    print("")