# Knowledge
1. Traditional arbitrage is working: (Also proven in another project)
2. The essay (strategy_framework.tex) is not working on it's own, though it is a good reference for the ideas, it needs some modifications to work properly.

# New Idea:
Combination of statistical arbitrage and traditional arbitrage. Try to find traditional arbitrage then use xgboost **TO TRADE THE MOST PROFITABLE ONES**

# Constrain:
1. Exercise of options in China is pretty complex, we should avoid it
2. Modify initial capital to 100,000 RMB

# Plan:
Traditional arbitrage improve plan:

1. Use ideas from src-myoldproject (for reference only, notice it lack some method) to list avaliable arbitrage of that second (requirements: only those that have income > 0 at maturity)
2. Use the first part of strategy_framework to identify those that are most likely cashed in 5 minutes, instead of those best at holding to maturity
3. Note: current model and backtest are trained using very small subset, increase training effort, use optimization in backtest.
