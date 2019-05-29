class MaxHeap():
    global mapping, lemon_tree, ranking, normalizaition, weights, to_update

    # The parameter value must be a tuple (index, value)
    def __init__ (self):
        self.mapping = {}
        self.lemon_tree = []
        self.ranking = []
        self.normalizaition = []
        self.weights = []
        self.to_update = []

    def init_alpha(self, replay_memory_size, alpha):
        addition = 0
        for i in range(1, replay_memory_size + 1):
            self.ranking.append(pow(i, -alpha))
            self.weights.append(alpha)
            addition = addition + self.ranking[i-1]
            self.normalizaition.append(addition)

    def update_weights(self, beta):
        for i in range(0, len(self.weights)):
            self.weights[i] = pow(self.ranking[i] * len(self.lemon_tree), -beta)

    def insert(self, value):
        self.lemon_tree.append(value)
        self.mapping[value[0]] = len(self.lemon_tree) - 1
        end = True
        inserted_node = len(self.lemon_tree) - 1
        while(end and inserted_node != 0):
            parent = int(inserted_node / 2)
            if self.lemon_tree[inserted_node][1] > self.lemon_tree[parent][1]:
                self.swap(inserted_node, parent)
                inserted_node = parent
            else:
                end = False


    def swap(self, node, parent):
        temp = self.lemon_tree[node]
        self.lemon_tree[node] = self.lemon_tree[parent]
        self.lemon_tree[parent] = temp
        self.mapping[self.lemon_tree[parent][0]] = parent
        self.mapping[self.lemon_tree[node][0]] = node


    def update(self, value):

        lemon = self.mapping[value[0]] # Get the element from the mapping
        if value[1] < self.lemon_tree[lemon][1]:
            self.lemon_tree[lemon] = value # Update its value which is in the second position of value
            #self.restoreHeap(lemon, self.lemon_tree)
        if value[1] > self.lemon_tree[lemon][1]:
            self.lemon_tree[lemon] = value # Update its value which is in the second position of value
            self.push_up(lemon)

    def push_up(self, position):
        end = True
        while(end):
            parent = int(position / 2)
            if self.lemon_tree[position][1] > self.lemon_tree[parent][1]:
                self.swap(position, parent)
                position = parent
            else:
                end = False
            if position == 0:
                end = False

    def restoreHeap(self, i, heap):
        end = True
        while(end):
            largest = i
            r_child = int(2 * i + 1)
            l_child = int(2 * i)
            if l_child < len(heap) and heap[l_child][1] > heap[largest][1]:
                largest = l_child

            if r_child < len(heap) and heap[r_child][1] > heap[largest][1]:
                largest = r_child

            if largest != i:
                heap[largest], heap[i] = heap[i], heap[largest]
                self.mapping[heap[largest][0]] = largest
                self.mapping[heap[i][0]] = i
                i = largest
            else:
                end = False



    def pop(self, heap):
        root = heap[0]
        tail = heap.pop()
        if len(heap) == 0:
            heap.append(tail)
        heap[0] = tail
        self.restoreHeap(0, heap)
        return root

    def delete(self, key):
        diff = False
        to_delete = self.mapping[key]
        if self.lemon_tree[to_delete][0] != key:
            print("hoibo")
        if self.lemon_tree[to_delete] != self.lemon_tree[-1]:
            diff = True
        self.lemon_tree[to_delete], self.lemon_tree[-1] = self.lemon_tree[-1], self.lemon_tree[to_delete]
        self.mapping[self.lemon_tree[to_delete][0]] = to_delete
        self.lemon_tree.pop()
        del self.mapping[key]
        if diff:
            self.restoreHeap(to_delete, self.lemon_tree)

    def get_N(self, n):
        heap = []
        for lemon in self.lemon_tree:
            heap.append(lemon)
        sorted_array = []
        for i in range(0, n):
            sorted_array.append(self.pop(heap))
        return sorted_array

    def all(self):
        return self.get_N(len(self.lemon_tree)-1)
    def sort(self):

        self.lemon_tree = self.get_N(len(self.lemon_tree))
        i = 0
        for lemon in self.lemon_tree:
            self.mapping[lemon[0]] = i
            i = i + 1

    def logical_update(self, value):
        self.to_update.append(value)

    def update_all(self):
        while len(self.to_update) > 0:
            element = self.to_update.pop()
            if element[0] in self.mapping:
                self.update(element)
