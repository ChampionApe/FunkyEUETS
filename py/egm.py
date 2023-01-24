import numpy as np
from numba import jit, float64, int8
from numba.experimental import jitclass
from numba.typed import Dict

@jitclass([ ('α', float64),
			('γ', float64),
			('ψ', float64),
			('δ', float64),
			('κ', float64),
			('β', float64),
			('g0',float64),
			('pf', float64),
			('stepSize', float64),
			('T',int8),
			('globalMaxB',float64)])
class EGM:
	""" Given parameter values """
	def __init__(self, α=.186, γ=710.0, ψ=1.0, δ=5.0, g0=0.006, κ=0.01, β=1/1.05, pf = 40.5, T = 95, stepSize = 0.01,globalMaxB = 20.0):
		self.α, self.γ, self.ψ, self.δ, self.g0, self.κ, self.β, self.pf, self.T, self.stepSize, self.globalMaxB = α,γ,ψ,δ,g0,κ,β,pf,T,stepSize,globalMaxB

	def technologyIndex(self, A0=1):
		A,g = np.empty(self.T), np.empty(self.T)
		g[0], A[0] = self.g0, A0
		for i in range(1,self.T):
			g[i] = g[i-1]/(1+self.κ)
			A[i] = A[i-1]*(1-g[i])
		return A

	def price(self, A, e):
		return A*(self.α*self.γ*(e+self.ψ)**(self.α-1)-self.δ)-self.pf

	def quant(self, A, p):
		return (A*self.α*self.γ/(A*self.δ+p+self.pf))**(1/(1-self.α))-self.ψ

	def solveTerminal(self, A, y):
		b = np.arange(y, y+self.quant(A,0), self.stepSize)
		return b, b+y, self.price(A, b+y)

	def solve_t(self, bLead, eLead, pLead, A, y):
		b,e,p = None, None, None
		eint = self.quant(A, self.β*pLead)
		bint = bLead-y+eint
		keepint = ((bint>0) & (bint<self.globalMaxB))
		if bint[0]<=0:
			i = len(bint[bint<0])
			x = bLead[i+1]-bint[i+1] * (bLead[i+1]-bLead[i])/(bint[i+1]-bint[i])
			b = np.append(0, bint[keepint])
			e = np.append(y-x, eint[keepint])
		else:
			b = np.append(np.arange(0, bint[0], self.stepSize), bint[keepint])
			e = np.append(np.arange(0, bint[0], self.stepSize)+y, eint[keepint])
		p = self.price(A,e)
		return b,e,p

	def solve(self, A, y):
		b,e,p = Dict(), Dict(), Dict()
		t=self.T-1
		b[t], e[t], p[t] = self.solveTerminal(A[t],y[t])
		for t in range(self.T-2,-1,-1):
			b[t],e[t],p[t] = self.solve_t(b[t+1],e[t+1],p[t+1],A[t],y[t])
		return b,e,p

	def sim(self, bgrid, egrid, pgrid, A, y, b0):
		b,e,p = np.empty(self.T+1), np.empty(self.T), np.empty(self.T)
		b[0]  = b0
		for t in range(0, self.T):
			e[t]  = np.interp(b[t], bgrid[t], egrid[t])
			p[t]  = self.price(A[t],e[t])
			b[t+1]= b[t]+y[t]-e[t]
		return b,e,p

@jit
def funcDummyUpper1(B, z, ξ):
	return 1 if (B>0.833 and z>=0.12*ξ*B) else 0
@jit
def funcDummyUpper2(B, z, ξ):
	return 1 if (B>0.833 and z<0.12*ξ*B) else 0
@jit
def funcDummyLower1(B, M):
	return 1 if (B<0.4 and M>=0.1) else 0
@jit
def funcDummyLower2(B, M):
	return 1 if (B<0.4 and M<0.1) else 0
@jit
def funcNA(BL, zL, ξL, ML, z):
	return 0.12*ξL*BL*funcDummyUpper1(BL,zL,ξL)+z*funcDummyUpper2(BL,zL,ξL)-0.1*funcDummyLower1(BL,ML)-ML*funcDummyLower2(BL,ML)
@jit
def funclawOfMotionM(ML, NA, zL, NAL):
	return min(ML+NA, max(0, 0.57*zL-NAL))
@jit
def ETSSimple(M0, B, z, ξ, NA0 = 0):
	M, NA, y = np.empty(len(z)), np.empty(len(z)), np.empty(len(z))
	M[0], NA[0], y[0] = M0, NA0, z[0]-NA0
	for t in range(1,len(z)):
		NA[t] = funcNA(B[t-1], z[t-1], ξ[t-1], M[t-1], z[t])
		y[t]  = z[t]-NA[t]
		M[t]  = M[t-1]+NA[t]
	return M, NA, y
@jit
def ETS(M0,B,z,ξ,NA0 = 0):
	M, NA, y = np.empty(len(z)), np.empty(len(z)), np.empty(len(z))
	M[0], NA[0], y[0] = M0, NA0, z[0]-NA0
	for t in range(1,len(z)):
		NA[t] = funcNA(B[t-1], z[t-1], ξ[t-1], M[t-1], z[t])
		y[t]  = z[t]-NA[t]
		M[t]  = funclawOfMotionM(M[t-1], NA[t], z[t-1],NA[t-1])
	return M, NA, y
