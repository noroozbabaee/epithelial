import numpy as np

def foo1(i, GAM):
    x = GAM[0]
    y = GAM[1]
    z = GAM[2]
    fun0 = 2 * x ** 2 + z
    fun1 = 2 * x + y ** 3 - z
    fun2 = 0.3 * x - 3 * y + 4 * z
    fun = [fun0, fun1, fun2]
    return fun[i]


def Matrix(init, w, h):
    return [[init for x in range(h)] for y in range(w)]


# def calculate(X, T):
dif = Matrix(0, 3, 3)
funi = []
funf = []
fun_i = []
fun_f = []
dif = []
dif_f = []
n = 10**-9
Del = [1, 1, 1]
Del = np.asarray(Del) * n
Del = np.array(Del)
GAM = [1,2,4]
# [a+b for a, b in zip(GAM, Del)]
GAMM = []
GAM_M = []


# print(foo1(0,GAM))
for i in range(0,3):

    fun_i = float(foo1(i, GAM))
    funi.append(fun_i)
#    print("funi", funi)
# for k,_in enumerate(3):
for k in range(0,3):
 #   print(Del[k])
    #print(GAM[k])
    GAM[k]= np.add(Del[k], GAM[k])
    GAM_M.append(GAM)
  #  print("GAM", GAM_M)
    for j in range(0,3):
        # fun_i = float(foo1(j, GAM))
        # print("fun_i", fun_i)
        # funi.append(fun_i)
        fun_f = float(foo1(j, GAM))
        funf.append(fun_f)
#        print("funf", funf)
        # print("%f %f"%(fun_i, Del[0]))
#         dif_f = float(fun_f - fun_i) / Del[0]
#         dif.append(dif_f)
#         print("dif_f", dif)
#     # Del_f = fun_f - fun_i
#     # print(Del_f)
#     # DERIV ='{0:.50f}'.format(Del_f/Del[0])
# #print(type(dif))
# dif = np.array(dif)
# #print(type(dif))
#
# dif = np.reshape(dif, [3, 3])
#print("dif", dif)
# funi = np.array(funi)
# funf = np.array(funf)
# np.set_printoptions(suppress=True)
# print("funf", funf)
# print(type(funf))

funi=[funi ,funi, funi ]
print('funi=',funi)
funf = np.reshape(funf, [3, 3])
print('funf=',funf)
dif = np.subtract(funf,funi)
print('dif =',dif)





# print(funf.strip(' '))
# str ="testing, 1, 2 ,3"
# # print(str.split(","))
# # print(np.array(str.split(" ")))
# Deriv = (np.array(funf[:]) - np.array(funi[:]))/Del[0]
# type(Deriv)
# Deriv=np.reshape(Deriv, [3,3])

# a = [1, 2, 3]
# b = a
# a[:] = [x + 2 for x in a]
# print(b)
# from sympy import diff, symbols,lambdify,sin,cos
#
# def my_eqs(x,y,z):
#     x,y,z= symbols('x y z')
#     fun1 = x + z
#     fun2 = 2 * x + y - z
#     fun3 = 0.3 * x - 3 * y + z
#     d_fun1x=diff(fun1,x)
#     d_fun1y=diff(fun1,y)
#     d_fun1z=diff(fun1,z)
#     d_fun2x = diff(fun2, x)
#     d_fun2y = diff(fun2, y)
#     d_fun2z = diff(fun2, z)
#     d_fun3x = diff(fun3, x)
#     d_fun3y = diff(fun3, y)
#     d_fun3z = diff(fun3, z)
#     d_fun = [[d_fun1x, d_fun1y, d_fun1z],[d_fun2x, d_fun2y, d_fun2z],[d_fun3x, d_fun3y, d_fun3z]]
#     return d_fun
#
# print('d_f',my_eqs(2,1,-2))

# from sympy import diff, symbols,lambdify,sin,cos
# x = symbols('x')
# y = symbols('y')
# expr = sin(x) + cos(x) * y
# print('expr',expr)
#
# f = lambdify([x,y], expr, 'numpy')
# a = np.array([0,1,2])
# b = np.array([1])
# print('f',f(a,b))