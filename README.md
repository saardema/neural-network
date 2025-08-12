# Neural Network
## Features
- Multi threading
- Stochastic gradient descent
- Mini batched training
- Visual representation of the network
- Realtime data visualization of the input and output
- Customizable and configurable

GPU based learning is planned and in progress


## Glossary
### Terms
| Term       | Description                                                              |
| ---------- | ------------------------------------------------------------------------ |
| Epoch      | One full pass of the training set                                        |
| Batch      | A parameter adjustment based on the average of multiple training samples |
| Batch step | Individual training sample evaluation added to the average sum           |

### Batch steps
```
for each batch step:
	for each layer:
		for each neuron:
			compute activation # 1
	compute output error
	for each layer backwards:
		for each neuron:
			for each next_neuron:
				compute delta # 2
			for each weight:
				increment gradient # 3
			increment bias gradient # 3

for each layer:
	for each neuron:
		increment weights by average weight gradient # 4
		increment bias by average bias gradient # 4
```
