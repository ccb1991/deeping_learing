"""
Created by 陈辰柄 
datetime:2020/2/23 12:14
Describe: 乘法层和加法层
"""


class Mullayer:
    """乘法层"""

    def __init__(self):
        self.x = None
        self.y = None

    def forword(self, x, y):
        """向前传播"""
        self.x = x
        self.y = y
        return self.x * self.y

    def backward(self, dout):
        """反向传播"""
        dx = self.y * dout
        dy = self.x * dout
        return dx, dy


class AddLayer:
    """加法层"""

    def __init__(self):
        pass

    def forward(self, x, y):
        return x + y

    def backward(self, dout):
        dy = dx = dout * 1
        return dx, dy


if __name__ == '__main__':
    # 乘法层
    apple = 100
    apple_num = 2
    tax = 1.1

    mul_apple_layer = Mullayer()
    mul_tax_layer = Mullayer()

    # 向前传播
    apple_price = mul_apple_layer.forword(apple, apple_num)
    price = mul_tax_layer.forword(apple_price, tax)

    print(price)

    # 反向传播
    apple_price, apple_tax = mul_tax_layer.backward(1)
    # 因为正想传播的时候apple单价为x，个数为y所以求导时对x求导就是apple的单价导数
    one_apple_price,apple_num=mul_apple_layer.backward(apple_price)
    print(apple_num,one_apple_price,apple_tax)
