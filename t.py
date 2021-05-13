class A():
    def __init__(self, num):
        self.num = self.b(num)
        self.a = None
    def b(self, a):
        return a + 2
    def print_num(self):
        print(self.num)

class B(A):
    def __init__(self, a):
        super().__init__(a)
        self.a = 1

    def b(self, a):
        return a + 3

a = B(10)
a.print_num()
print(a.a)