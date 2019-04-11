import torch

# hand-implemented queue storing image data of shape 3*28*28
class ImageQueue():
    def __init__(self, maxsize):
        self.ttype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.tttype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
        self.arr = torch.zeros((maxsize, 3, 28, 28)).type(self.ttype)
        self.i = 0
        self.maxsize = maxsize
        self.uplimit = self.i
    def enqueue(self, x):
        assert(self.i+len(x) <= self.maxsize) # lazy
        self.arr[self.i:(self.i+len(x))] = x
        self.i = (self.i + len(x)) % self.maxsize
        if self.uplimit < self.maxsize:
            self.uplimit += min(len(x), self.maxsize)
    def sample(self, num):
        idx = torch.randint(0, self.uplimit, (num,)).type(self.tttype)
        return self.arr[idx]


# test
if __name__ == '__main__':
    q = ImageQueue(20)
    data = torch.randn(20,3*28*28)
    q.enqueue(data[:5])
    print(q.sample(5))
    q.enqueue(data[5:10])
    q.enqueue(data[10:15])
    q.enqueue(data[15:20])
    q.enqueue(data[5:10])
    assert(torch.all(q.arr[:5] == q.arr[5:10]))
