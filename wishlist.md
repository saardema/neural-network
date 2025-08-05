# Wishlist

## Network
- ### Multi-threaded mini-batch training
	#### Steps
	Each thread
	- performs forward activation pass from unique sample
	- computes its own output error
	- computes its own deltas
	- increments the central gradients

	Main thread
	-	Applies gradients and resets them

	#### Data structure
	Each thread needs to write to its own
	- activations per neuron
	- deltas per neuron

## Plotter
### Time graph
- Support longer time range than 10s

### Dots & Lines
- Show more than one plot


## Glossary
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
