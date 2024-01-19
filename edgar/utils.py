from collections import defaultdict



NA_Dict = {'A': 1, 'G': 2, 'C': 3, 'U': 4, 'T': 5}


class NaStack:
    inv_dict = {')':'(', ']':'[', '}':'{', '>':'<', 'a':'A', 
                'b':'B', 'c':'C', 'd':'D', 'e':'E', 'f':'F'}
    
    def __init__(self):
        self.st = defaultdict(list)

    def __setitem__(self, k, v):
        self.st[k].append(v)
    
    def __getitem__(self, k):
        if len(self.st[self.inv_dict[k]]):
            return self.st[self.inv_dict[k]].pop()
        return None
        
    def isempty(self):
        for k in self.st:
            if len(self.st[k])!=0:
                return False
        return True
    

def get_na_pairs(struct):
    pairs = []
    stack = NaStack()

    for i, s in enumerate(struct):
        if s == '.':
            continue

        if s in '([{<ABCDEF':
            stack[s] = i

        elif s in ')]}>abcdef':
            op_idx = stack[s]

            if op_idx is None:
                raise ValueError("Structure has a pair without an opening bracket")
                    
            pairs.append((op_idx, i))

    if not stack.isempty():
        raise ValueError("Structure has a pair without a closing bracket")
        
    return pairs