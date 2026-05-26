# Theoretical Foundations and Strategic Motivation

## 1. Mathematical Definitions

### Total Positivity (TP)
A kernel $K: \mathcal{X}\times\mathcal{Y}\to\mathbb{R}_+$ is **$TP_r$** if for all $n\le r$ and ordered points $x_1<\dots<x_n$, $y_1<\dots<y_n$: 

$$ \det\!\big[K(x_i,y_j)\big]_{i,j=1}^n \;\ge\; 0. $$ 

$TP_\infty$ holds for all $n$. For smooth $K>0$, $TP_2 \iff \partial_x\partial_y\ln K \ge 0$. 

### Black–Scholes Formula
For spot $S$, strike $K$, maturity $T$, rate $r$, vol $\sigma$: 

$$ \begin{aligned}
C(S,K,T) &= S\,\Phi(d_1) - K e^{-rT}\Phi(d_2),\\
P(S,K,T) &= K e^{-rT}\Phi(-d_2) - S\,\Phi(-d_1),\\
d_{1,2} &= \frac{\ln(S/K)+(r\pm\tfrac12\sigma^2)T}{\sigma\sqrt{T}}.
\end{aligned} $$

---

### Example
*Parameters: $S_0=100$, $r=5\%$, $\sigma=20\%$*  
*Strikes $K$: 80, 85, 90, 95, 100, 105, 110, 115, 120*  
*Time to Expiry $T$: 10, 40, 70, 110 days*

### European Call Prices $C(S_0,K,T)$
| K | T = 10 | T = 40 | T = 70 | T = 110 |
|---|--------|--------|--------|---------|
| **120** | 0.00 | 0.01 | 0.09 | 0.33 |
| **115** | 0.00 | 0.06 | 0.28 | 0.75 |
| **110** | 0.00 | 0.28 | 0.79 | 1.55 |
| **105** | 0.12 | 1.04 | 1.91 | 2.95 |
| **100** | 1.39 | 2.92 | 3.97 | 5.13 |
| **95**  | 5.21 | 6.22 | 7.13 | 8.20 |
| **90**  | 10.12| 10.61| 11.23| 12.06|
| **85**  | 15.12| 15.48| 15.89| 16.50|
| **80**  | 20.11| 20.44| 20.77| 21.25|

### European Put Prices $P(S_0,K,T)$
| K | T = 10 | T = 40 | T = 70 | T = 110 |
|---|--------|--------|--------|---------|
| **80**  | 0.00 | 0.00 | 0.01 | 0.05 |
| **85**  | 0.00 | 0.01 | 0.07 | 0.22 |
| **90**  | 0.00 | 0.12 | 0.37 | 0.71 |
| **95**  | 0.08 | 0.70 | 1.22 | 1.78 |
| **100** | 1.25 | 2.37 | 3.02 | 3.64 |
| **105** | 4.97 | 5.47 | 5.91 | 6.38 |
| **110** | 9.85 | 9.68 | 9.74 | 9.90 |
| **115** | 14.84| 14.43| 14.18| 14.03|
| **120** | 19.84| 19.35| 18.94| 18.54|

**Notes:**
- Prices in currency units, rounded to 2 decimals; values $<0.005$ shown as 0.00.
- For fixed $K$, call prices $\uparrow$ in $T$ (time value); put prices generally $\uparrow$ in $T$ for OTM/ATM strikes.
- For fixed $T$, call prices $\downarrow$ in $K$; put prices $\uparrow$ in $K$, consistent with no-arbitrage monotonicity implied by TP structure.
- Both matrix follow TP2 after considered time value of money

---

## 2. Proof of Total Positivity in Option Prices

**Lemma: Total Positivity of the Lognormal Kernel**  
The standard normal density $\phi(z) = \frac{1}{\sqrt{2\pi}}e^{-z^2/2}$ is a **Polya frequency function of infinite order** ($PF_\infty$), hence $TP_\infty$ (Karlin, 1968). Since $TP_\infty$ is preserved under:
- strictly monotone transformations of arguments,
- multiplication by positive functions of a single variable,

the lognormal density $p_T(S,x)$ is $TP_\infty$ on $(0,\infty) \times (0,\infty)$ in $(S,x)$.

**Proof: Payoff Kernel and Composition**  
Define the payoff kernel $\psi(x,K) = (x-K)^+$. This kernel is $TP_2$ in $(x,K)$ because for $x_1<x_2$, $K_1<K_2$, 

$$ \det \begin{pmatrix} (x_1-K_1)^+ & (x_1-K_2)^+ \\ (x_2-K_1)^+ & (x_2-K_2)^+ \end{pmatrix} \ge 0, $$ 

which follows from the single-crossing property of call payoffs. By **Karlin’s Basic Composition Formula** (1968, Thm. 3.1), if $p_T(S,x)$ is $TP_\infty$ and $\psi(x,K)$ is $TP_2$, then the integral transform 

$$ C(S,K) = e^{-rT} \int_0^\infty \psi(x,K) p_T(S,x) \, dx $$ 

is $TP_2$ in $(S,K)$. 

**Mathematical Statement**  
For all $0 < S_1 < S_2$ and $0 < K_1 < K_2$, 

$$ C(S_1,K_1) C(S_2,K_2) \ge C(S_1,K_2) C(S_2,K_1). $$ 

Equivalently, in regions where $C$ is smooth and positive, 

$$ \frac{\partial^2}{\partial S \partial K} \ln C(S,K) \ge 0. $$ 

This implies $\partial_K \big( C_S / C \big) \le 0$, i.e., the elasticity-adjusted delta is non-increasing in strike.

---

## Note: Empirical Motivation & Strategy Pivot

During research, it was found that the findings from [SSRN Paper 5392317](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5392317) could not be reproduced in the A-share market. A workaround was needed. 

Furthermore, as the original paper was written from a less rigorous perspective, we can leverage advanced quantitative methods. Instead of standard approaches, we directly study $TP_n$ on the A-share option matrix (which is $4 \times 9$ in dimension). 

**Core Idea:** 
- Calculate the deviation rate of each market point from the theoretical $TP$ matrix surface. 
- Study the regression characteristics 1 to 10 minutes later (rather than 1 day later) to maximize profit potential.

**Bridge to the Strategy Framework:**  
The total-positivity analysis above gives a practical way to turn each live option matrix into trading signals. For every tick, we compare the observed bid--ask quotes with a theoretical TP-consistent matrix surface. The distance from this surface tells us whether a contract is likely underpriced, overpriced, or too close to fair value to trade.

The next step is developed in `strategy_framework.tex`. It starts from a signal matrix $w_{ij}$ and explains how to convert those signals into executable portfolio weights. The workflow is:

1. **Matrix Translation:** Convert each $4 \times 15$ option quote matrix into a signal matrix $w_{ij}$, where each entry measures the expected short-term regression tendency.
   - If $w_{ij} > 0$, the market ask is below the TP surface, so the option appears underpriced and may be bought.
   - If $w_{ij} < 0$, the market bid is above the TP surface, so the option appears overpriced and may be sold.
   - If the TP surface lies inside the bid--ask spread, set $w_{ij} = 0$ because the edge is not tradable.
   - This translation must remain lightweight: no giant models, and the target latency is under 10ms per tick.
2. **Optimization and Execution:** Use $w_{ij}$ together with liquidity, risk limits, and transaction costs to choose portfolio weights and send orders.
