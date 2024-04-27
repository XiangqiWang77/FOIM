from scipy import stats

# Sample data for two groups
#group1 = [9.1, 21.085, 1.44, 10.42, 27.25, 4.05, 5.08, 1.87, 44.27, 2.95, 4.05, 2.61]  # Sample data for group 1
group1=[12530.01, 196292.60, 607.64, 14720.88, 240386.97, 1117.88, 7241.75, 21944.96, 14011.87, 136.83, 1747.43, 662.76]
#group2 = [0.39, 0.11, 1.17, 0.61, 0.33, 2.01, 0.12, 0.50, 32.33, 0.08, 0.07, 1.33]  # Sample data for group 2
group2=[244.02, 3556.61, 1073.29,4886, 44893.83, 931.73, 133.6, 552.86, 10548.99, 49.82, 742.39, 744.18]

# Perform t-test
t_statistic, p_value = stats.ttest_ind(group1, group2)

# Determine significance level
alpha = 0.15

# Print results
print("T-Statistic:", t_statistic)
print("P-Value:", p_value)

if p_value < alpha:
    print("Reject the null hypothesis. There is a significant difference between the two groups.")
    if t_statistic < 0:
        print("Group 1 mean is significantly lower than Group 2 mean.")
    else:
        print("Group 2 mean is significantly lower than Group 1 mean.")
else:
    print("Fail to reject the null hypothesis. There is no significant difference between the two groups.")