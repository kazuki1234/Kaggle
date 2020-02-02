from sklearn.model_selection import KFold

# Cross validation
kf = KFold(n_split=4, shuffle=True, random_state=71)
for i, (tr_idx, va_idx) in enmuerate(kf.split(train_x):

    # Split into training and validation data
    tr_x, va_x = train_x.iloc[tr_idx].copy(), train_x.iloc[va_idx].copy()
    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx] 

    # Make target encoding for each columns
    for c in cat_cols:
        # Make target mean for validation data
        data_tmp = pd.DataFrame([c: tr_x[c], 'target': tr_y])
        target_mean = data_tmp.groupby(c)['target'].mean()
        va_x.loc[:, c] = va_x[c].map(target_mean)

        # Folds for target encoding
        tmp = np.repeat(np.nan, tr_x.shape[0])
        kf_encoding = KFold(n_split=4, shuffle=True, random_state=72)
        for idx_1, idx_2 in kf_encoding.split(tr_x):
            target_mean = data_tmp.iloc[idx_1].groupby(c)['target'].mean()
            tmp[idx_2] = tr_x[c].iloc[idx_2].map(target_mean)

        tr_x.loc[:,c] = tmp

