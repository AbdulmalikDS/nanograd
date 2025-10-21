## أمثلة لبعض الاشتقاقات

```
( f(x-h) - f(x) ) / h
( (d+h) * f - d * f ) / h
( (d * f + h * f - d * f) / h
( ( h * f ) / h )
```

## dL / dc عشان نشتق

لازم قبلها نشتق
dd / dc

```
d = c + e

( (c + h + e) - (c + e) ) / h
( c + h + e - c - e) / h
```

نشوف انه اللي يبقى فقط
```
h/h
```
ويصير الناتج 1

## dd / de ونفس الشيء ل

c الى اخر عصبون اللي هو L وعشان نسوي الاشتقاق من

Chain rule لازم نسوي ال

```
dL / dc = (dL / dd) * (dd / dc)
```

الناتج يصير 1.0 * -2.0

## ملاحظة مهمة

the plus op is just routing the local gradient value

---

## التعليقات الأخرى في الكود

في ملف `examples/neuron_demo.py` - التطبيق الموسع لعملية tanh:

```python
# e = (2*n).exp()              # exponential
# o = (e - 1) / (e + 1)        # tanh formula
```

