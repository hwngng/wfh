let adjs = new Map([
		['1', {
			'p': new Set([]),
			'c': new Set(['3', '2'])
		}],
		['3', {
			'p': new Set([]),
			'c': new Set(['4', '5'])
		}],
		['2', {
			'p': new Set([]),
			'c': new Set(['6', '7'])
		}],
		['7', {
			'p': new Set([]),
			'c': new Set(['3', '5'])
		}]
	])
let edges = new Map([['3', 0]])
let queue = ['3']
while (queue.length > 0) {
	s = queue[0]
	queue.shift()
	adjs.get(s)?.c.forEach((adjacent,i) => {
		if (!edges.has(adjacent))
		{
			edges.set(adjacent, edges.get(s) + 1)
			queue.push(adjacent);
		}
	});
}

console.log(edges)

let nodebyEdge = new Map()
for (item of edges) {
	if (!nodebyEdge.has(item[1])) {
		nodebyEdge.set(item[1], [])
	}
	nodebyEdge.get(item[1]).push(item[0])
}

console.log(nodebyEdge)