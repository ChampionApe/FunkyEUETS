import numpy as np
from numba import jit

_defaultParameters = { 'α': .186, 
						'γ': 710, 
						'ψ': 1, 
						'δ': 5, 
						'g0': 0.006, 
						'κ':0.01, 
						'β': 1/1.05, 
						'pf': 40.5, 
						'T':95, 
						'stepSize': 0.01,
						'globalMaxB': 20}

@jit
def technologyIndex(T, g0, κ, A0 = 1):
	A,g = np.empty(T), np.empty(T)
	g[0], A[0] = g0, A0
	for i in range(1,T):
		g[i] = g[i-1]/(1+κ)
		A[i] = A[i-1]*(1-g[i])
	return A

@jit
def price(A, e, α, γ, ψ, δ, pf):
	return A*(α*γ*(e+ψ)**(α-1)-δ)-pf

@jit 
def quant(A, p, α, γ, ψ, δ, pf):
	return (A*α*γ/(A*δ+p+pf))**(1/(1-α))-ψ

@jit
def solveTerminal(A, y, α, γ, ψ, δ, pf, stepSize):
	b = np.arange(y, y+quant(A, 0, α, γ, ψ, δ, pf), stepSize)
	return b, b+y, price(A, b+y, α, γ, ψ, δ, pf)

@jit
def solve_t(bLead, eLead, pLead, A, y, α, γ, ψ, δ, pf, β, globalMaxB, stepSize):
	eint = quant(A, β*pLead, α, γ, ψ, δ, pf)
	bint = bLead-y+eint
	keepint = ((bint>0) & (bint<globalMaxB))
	if bint[0]<=0:
		i = len(bint[bint<0])
		if bint[i+1]==bint[i]:
			x = bLead[i+1]
		else:
			x = bLead[i+1]-bint[i+1] * (bLead[i+1]-bLead[i])/(bint[i+1]-bint[i])
		b = np.append(0, bint[keepint])
		e = np.append(y-x, eint[keepint])
	else:
		b = np.append(np.arange(0, bint[0], stepSize), bint[keepint])
		e = np.append(np.arange(0, bint[0], stepSize)+y, eint[keepint])
	p = price(A, e,  α, γ, ψ, δ, pf)
	return b,e,p


class EGM:
	""" Given parameter values """
	def __init__(self, **kwargs):
		[self.__setattr__(k,kwargs[k] if k in kwargs else _defaultParameters[k]) for k in _defaultParameters]

	def technologyIndex(self, A0=1):
		return technologyIndex(self.T, self.g0, self.κ, A0=A0)

	def price(self, A, e):
		return price(A, e, self.α, self.γ, self.ψ, self.δ, self.pf)

	def quant(self, A, p):
		return quant(A, p, self.α, self.γ, self.ψ, self.δ, self.pf)

	def solveTerminal(self, A, y):
		return solveTerminal(A, y, self.α, self.γ, self.ψ, self.δ, self.pf, self.stepSize)

	def solve_t(self, bLead, eLead, pLead, A, y):
		return solve_t(bLead, eLead, pLead, A, y, self.α, self.γ, self.ψ, self.δ, self.pf, self.β, self.globalMaxB, self.stepSize)

	def solve(self, A, y):
		b,e,p = {}, {}, {}
		t = self.T-1
		b[t],e[t],p[t] = self.solveTerminal(A[t], y[t])
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

