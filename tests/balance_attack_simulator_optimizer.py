from scipy import stats
for i in range(1,20):
    n = 4 * i
    differ = stats.binom(n,0.5)
    upper_bound = int(2.5 * i)
    lower_bound = int(1.5 * i)
    print(1-differ.cdf(upper_bound)+differ.cdf(lower_bound),lower_bound,upper_bound,i)

