import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np 

def handle_df(df):
	mean = np.mean(df['mean_wt'].tolist(), axis = 1)
	df.insert(2, "mean", mean, True)
	conf_int = np.std(df['mean_wt'].tolist(), axis = 1, ddof = 1) * 1.96 / np.sqrt(len(df['mean_wt'].tolist()[0]))
	df.insert(2, "confint", conf_int, True)
	return df

def plot(df):
	n = df['servers'].unique()
	alpha = 0.6
	c = ['blue', 'red', 'orange']
	for i,servers in enumerate(n):

		df_n = df.loc[df['servers'] == servers]
		plt.errorbar(df_n['rho'], df_n['mean'], yerr = df_n['confint'], marker='.', fmt="."\
			, solid_capstyle="projecting", capsize=5, label='{} server(s)'.format(int(servers)))

def plot_together(df, df1):
	
	n = df['servers'].unique()
	alpha = 0.6
	for i,servers in enumerate(n):
		df_n = df.loc[df['servers'] == servers]
		plt.errorbar(df_n['rho'], df_n['mean'], yerr = df_n['confint'], marker='.', fmt="."\
			, solid_capstyle="projecting", capsize=5, label='{} server(s)'.format(int(servers)))

	df_n = pd.DataFrame()
	df_n = df1
	plt.errorbar(df_n['rho'], df_n['mean'], yerr = df_n['confint'], marker='.', fmt="."\
		, solid_capstyle="projecting", capsize=5, label='{} server(s), priority'.format(1))

# 124 M
f124M = "M124-p1000-t300-sims30_2.csv"

# 1 M prio
f1Mp = "M124-p1000-t300-sims30-prio_2.csv"

# 124D
f124D = "D124-p1000-t300-sims30_2.csv"

# 124LT
f124LT = "LT124-p1000-t300-sims30_2.csv"


df1 = pd.read_csv(f124M, converters={'mean_wt': eval})
df2 = pd.read_csv(f1Mp, converters={'mean_wt': eval})

plt.figure()
df1 = handle_df(df1)
df2 = handle_df(df2)
plot_together(df1, df2)
plt.legend(fontsize=16)
plt.xlabel("rho [-]", fontsize = 16)
plt.ylabel("mean waiting time [-]", fontsize = 16)

plt.figure()
df = pd.read_csv(f124LT, converters={'mean_wt': eval})
df1 = handle_df(df)
plot(df1)
plt.legend(fontsize=16)
plt.xlabel("rho [-]", fontsize = 16)
plt.ylabel("mean waiting time [-]", fontsize = 16)

plt.figure()
df = pd.read_csv(f124LT, converters={'mean_wt': eval})
df1 = handle_df(df)
plot(df1)

plt.xlabel("rho [-]", fontsize = 16)
plt.ylabel("mean waiting time [-]", fontsize = 16)
plt.legend(fontsize=16)
plt.show()