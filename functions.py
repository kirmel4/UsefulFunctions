def gini_score_mine(y_true: pd.Series, y_pred: pd.Series) -> float:
    if len(y_true.unique()) == 2:
        return 2*roc_auc_score(y_true, y_pred) - 1
    elif len(y_true.unique())>2:
        gini_multiclass = 0
        for label in y_true.unique():
            y_pred_temp = [(y==label) for y in y_pred]
            y_true_temp = [(y==label) for y in y_true]
            gini_multiclass += sum(y_true_temp)*(2*roc_auc_score(y) - 1)
    else:
        print('Not computable')
        return gini_multiclass / y_true.size

def permutation_importance(model , data: pd.DataFrame, features_list: list, score_name: str, target_name: str, n_times: int, func, greater_means_better: bool, classification: bool) -> pd.Series:
    PI_dict = {}
    score = func(data[target_name], data[score_name])
    data_copy = data.copy()
    if classification:
        data[score_name] = model.predict_proba(data[features_list])[:,1]
    else:
        data[score_name] = model.predict(data[features_list])
    for i, feature in tqdm(enumerate(features_list)):
        permuted_score = 0
        feature_copy = data_copy[feature]
        for j in range(n_times):
            np.random.seed(i+j)
            shuffled_feature = np.random.permutation(feature_copy)
            data_copy[f'{feature}'] = shuffled_feature
            model.predict_proba(data_copy[features_list])
            permuted_score += func(data_copy[target_name], model.predict_proba(data_copy[features_list])[:,1])
            data_copy[feature] = feature_copy
            print(permuted_score, feature)
        if greater_means_better:
            PI_dict[feature] = score  -  permuted_score/n_times
        else:
            PI_dict[feature] = func(data[target_name], data[score_name])  +  permuted_scores/n_times
    return pd.DataFrame(PI_dict.items(), columns = ['Feature', 'Permutation importance']).sort_values('Permutation importance')

def uplift(model, all_data: dict, permutation_importance_list: list, target_name: str, eval_set = None):
    uplift_features = []
    uplift_score_train = []
    uplift_score_valid = []
    uplift_score_oos = []
    uplift_score_oot = []
    x = np.linspace(0, len(permutation_importance_list))
    y = np.linspace(0, 1, 100)
    plt.ion()
    figure, ax = plt.subplots(figsize=(10, 8))
    line1, = ax.plot(x, y)
    plt.title("Geeks For Geeks", fontsize=20)
 
    # setting x-axis label and y-axis label
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    for feature in permutation_importance_list:
        uplift_features.append(feature)
        model.fit(train[uplift_features], train[target_name])
        new_y = model.predict_proba(all_data['train'][uplift_features])
        line1.set_xdata(x)
        line1.set_ydata(new_y)
        figure.canvas.draw()
        figure.canvas.flush_events()
        time.sleep(0.1)
        uplift_score_train.append(model.predict_proba(all_data['train'][uplift_features]))
        uplift_score_valid.append(model.predict_proba(all_data['valid'][uplift_features]))
        uplift_score_oos.append(model.predict_proba(all_data['oos'][uplift_features]))
        uplift_score_oot.append(model.predict_proba(all_data['oot'][uplift_features]))    

xgb_space={'max_depth': hp.quniform("max_depth", 1,8,1),
        'gamma': hp.uniform ('gamma', 1,20),
        'reg_alpha' : hp.quniform('reg_alpha', 1,100,1),
        'reg_lambda' : hp.uniform('reg_lambda', 0,1),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
        'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
        'n_estimators': 250
    }

def nan_filling(list_to_fill: list, alt_train, short_list):
for feature in list_to_fill:
    fill_model = LGBMRegressor(random_state=42, verbose = -1)
    fill_model.fit(alt_train[~alt_train[f"{feature}"].isna()][short_list].fillna(0), alt_train[~alt_train[f"{feature}"].isna()][f"{feature}"])
    pred = fill_model.predict(alt_train[alt_train[f"{feature}"].isna()][short_list].fillna(0)[short_list])
    alt_train.loc[alt_train[f"{feature}"].isna(), f"{feature}"] = pred

def selection(data, features, target_name):
    arr = []
    for feature in tqdm(features):
        clf = LGBMClassifier(n_estimators = 50, max_depth = 6, verbose =-1)
        clf.fit(pd.DataFrame(data[feature]), data[target_name])
        arr.append([gini_score_mine(data[target_name], clf.predict_proba(pd.DataFrame(data[feature]))[:,1]), feature])
    return arr

res_sorted = sorted(res, reverse=True)

