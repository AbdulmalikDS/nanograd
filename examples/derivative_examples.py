from engine import Value


def manual_derivative_example():
    """
    مثال على حساب الاشتقاق يدوياً باستخدام التعريف
    
    ( f(x-h) - f(x) ) / h
    """
    h = 0.0001
    
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a*b; e.label = 'e'
    d = e + c; d.label = 'd'
    f = Value(-2.0, label='f')
    L = d * f; L.label = 'L'
    L1 = L.data
    
    # نفس الحساب مع تغيير صغير
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    c.data += h  
    e = a*b; e.label = 'e'
    d = e + c; d.label = 'd'
    f = Value(-2.0, label='f')
    L = d * f; L.label = 'L'
    L2 = L.data
    
    manual_grad = (L2-L1)/h
    
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a*b; e.label = 'e'
    d = e + c; d.label = 'd'
    f = Value(-2.0, label='f')
    L = d * f; L.label = 'L'
    L.backward()
    
    print(f"Manual derivative dL/dc: {manual_grad}")
    print(f"Automatic derivative dL/dc: {c.grad}")
    print(f"Difference: {abs(manual_grad - c.grad)}")

if __name__ == "__main__":
    manual_derivative_example()

"""
شرح Chain Rule:

c الى اخر عصبون اللي هو L وعشان نسوي الاشتقاق من
Chain rule لازم نسوي ال

dL / dc = (dL / dd) * (dd / dc)

الناتج يصير 1.0 * -2.0 = -2.0

لأن:
- dL / dd = f = -2.0
- dd / dc = 1.0 (لأن d = e + c)

so the plus op is just routing the local gradient value
"""

