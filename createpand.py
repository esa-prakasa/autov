import pandas as pd




data = {'i':  [], 
'j': [], 
'r': [], 
'g': [], 
'b': [], 
}

for i in range(10):
	for j in range(10):
		r = i*2 + j
		g = i*3 + j
		b = i*5 + j

		data['i'].append(i)
		data['j'].append(i)
		data['r'].append(r)
		data['g'].append(g)
		data['b'].append(b)

df = pd.DataFrame (data, columns = ['i','j','r','g','b'])

print (df)