#distribution in our target attribute
sns.set(style="darkgrid")
plt.figure(figsize=(10,10))
sns.displot(df[config.TARGET_ATTR])
plt.savefig(os.path.join(config.OUTPUT_GRAPH_PATH,"target_count.png"))
plt.close()

#correlation
correlacio = df.corr()
plt.figure(figsize=(13,13))
ax = sns.heatmap(correlacio, annot=True, linewidths=.5)
ax.figure.savefig(os.path.join(config.OUTPUT_GRAPH_PATH,"corr.png"))
plt.close()

#correlation to target attributes
corr = df.corr()[config.TARGET_ATTR].sort_values(ascending=False)
plot = corr.plot(kind='bar')
fig = plot.get_figure()
fig.savefig(os.path.join(config.OUTPUT_GRAPH_PATH,"corr_to_target.png"))
plt.close()

#do scatter for most correlating attributes to target
best = corr[corr > config.THRESHOLD]
for i in range(1,len(best)):
    plt.scatter(df[best.index[i]], df[config.TARGET_ATTR])
    plt.xlabel(best.index[i])
    plt.ylabel(config.TARGET_ATTR)
    plt.savefig(os.path.join(config.OUTPUT_GRAPH_PATH, "scatter_"+best.index[i]+".png"))
    plt.close()