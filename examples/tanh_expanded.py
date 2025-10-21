from engine import Value

"""
مثال يوضح كيفية حساب tanh بطريقتين:
1. باستخدام الدالة المدمجة
2. بتوسيع المعادلة الرياضية

Example showing how to calculate tanh in two ways:
1. Using the built-in function
2. Expanding the mathematical equation
"""

x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')

w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')

b = Value(6.8813, label='b')

x1w1 = x1*w1; x1w1.label = 'x1w1'
x2w2 = x2*w2; x2w2.label = 'x2w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1w1x2w2'
n = x1w1x2w2 + b; n.label = 'n'

print("Using tanh function")
o1 = n.tanh(); o1.label = 'o'
o1.backward()
print(f"Output: {o1.data}")
print(f"n gradient: {n.grad}")

# الآن نفس الشيء لكن بتوسيع معادلة tanh
# tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)

print("Expanding tanh manually")
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
b = Value(6.8813, label='b')

x1w1 = x1*w1; x1w1.label = 'x1w1'
x2w2 = x2*w2; x2w2.label = 'x2w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1w1x2w2'
n = x1w1x2w2 + b; n.label = 'n'

e = (2*n).exp()              # exponential
o2 = (e - 1) / (e + 1)       # tanh formula
o2.label = 'o'
o2.backward()

print(f"Output: {o2.data}")
print(f"n gradient: {n.grad}")

print("Comparison")
print(f"Outputs match: {abs(o1.data - o2.data) < 1e-6}")
print(f"Gradients match: {abs(o1._prev.pop().grad - n.grad) < 1e-6}")

"""
ملاحظة:
o = tanh(n)
do/dn = 1 - tanh(n)^2
do/dn = 1 - o^2

مثال:
إذا كانت o = 0.7071
فإن do/dn = 1 - (0.7071)^2 = 1 - 0.5 = 0.5
"""

