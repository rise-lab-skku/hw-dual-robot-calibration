import numpy as np

def Rx(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]])

def Ry(b):
    cb, sb = np.cos(b), np.sin(b)
    return np.array([[cb,0,sb],[0,1,0],[-sb,0,cb]])

def Rz(g):
    cg, sg = np.cos(g), np.sin(g)
    return np.array([[cg,-sg,0],[sg,cg,0],[0,0,1]])

def derivs(alpha,beta,gamma):
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)
    dRx = np.array([[0,0,0],[0,-sa,-ca],[0,ca,-sa]])
    dRy = np.array([[-sb,0,cb],[0,0,0],[-cb,0,-sb]])
    dRz = np.array([[-sg,-cg,0],[cg,-sg,0],[0,0,0]])
    return dRx @ Ry(beta) @ Rz(gamma), Rx(alpha) @ dRy @ Rz(gamma), Rx(alpha) @ Ry(beta) @ dRz

# 예: 무작위 각도에서 유한차분 비교
a,b,g = 0.3, -0.4, 0.7
h = 1e-8
R0 = Rx(a) @ Ry(b) @ Rz(g)
Ra = Rx(a+h) @ Ry(b) @ Rz(g)
Rb = Rx(a) @ Ry(b+h) @ Rz(g)
Rg = Rx(a) @ Ry(b) @ Rz(g+h)
dRa, dRb, dRg = derivs(a,b,g)

print(np.max(np.abs(dRa - (Ra-R0)/h)))
print(np.max(np.abs(dRb - (Rb-R0)/h)))
print(np.max(np.abs(dRg - (Rg-R0)/h)))