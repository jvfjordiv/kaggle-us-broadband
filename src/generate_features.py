pd.set_option('display.max_columns', None)
df = pd.read_csv(config.INPUT_DATASET)

df["slowfrac_bbn"] = [element * 100 for element in df["slowfrac_bbn"]]
df["broad_cost"].fillna(0)
df["price_bbn"].fillna(0)

df = df.dropna()
df = df.loc[:, (df.dtypes != 'object')]
