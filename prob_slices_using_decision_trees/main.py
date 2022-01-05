#Importing required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

# data synthesizing
from ctgan import CTGANSynthesizer

import xgboost as xgb
from sklearn.tree import _tree
from xgboost import XGBClassifier
import pickle
import warnings
import json
import sys

def get_categorical_features(data, dataset_label_column_name):
    cat_feats = []
    for i, col in enumerate(data.dtypes):
        if np.issubdtype(col, np.integer) == False and np.issubdtype(col, np.floating) == False:
            if data.columns[i] == dataset_label_column_name:
                pass
            else:
                cat_feats.append(data.columns[i])
    print('Categorical features: ')
    print(cat_feats)
    return cat_feats


def create_tree_for_slices(x_val_df, y_val_df, xgbModel):
    y_predict = xgbModel.predict(x_val_df)
    y_predict_nparr = np.array(y_predict).flatten().reshape(-1,1)
    y_val_nparr = np.array(y_val_df).flatten().reshape(-1,1)
    is_correct = (y_predict_nparr == y_val_nparr) * 1
    dtc_labels = pd.DataFrame(is_correct)
    dtc = DecisionTreeClassifier(max_depth=2)
    dtc = dtc.fit(x_val_df, dtc_labels)

    return dtc


def get_dataframes_from_slices_by_tree(two_deep_tree, x_val_df, y_val_df):
    feature_names = x_val_df.columns

    tree_features = two_deep_tree.tree_.feature
    tree_threshold = two_deep_tree.tree_.threshold

    t_feature_names = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_features
    ]

    features_with_threshold = [(t_feature_names[i], tree_threshold[i]) for i in range(len(tree_features))]
    print(features_with_threshold)
    root = features_with_threshold[0]
    left = features_with_threshold[1]
    right = features_with_threshold[4]

    df1_filter = (x_val_df[root[0]] <= root[1])
    if left[0] != 'undefined!':
        df2_filter = (df1_filter & (x_val_df[left[0]] > left[1]))
        df1_filter = (df1_filter & (x_val_df[left[0]] <= left[1]))
    else:
        df2_filter = df1_filter
    
    df3_filter = (x_val_df[root[0]] > root[1])
    if right[0] != 'undefined!':
        df4_filter = (df3_filter & (x_val_df[right[0]] > right[1]))
        df3_filter = (df3_filter & (x_val_df[right[0]] <= right[1]))
    else:
        df4_filter = df3_filter

    x_df1 = x_val_df[df1_filter]
    x_df2 = x_val_df[df2_filter]
    x_df3 = x_val_df[df3_filter]
    x_df4 = x_val_df[df4_filter]

    y_df1 = y_val_df[df1_filter]
    y_df2 = y_val_df[df2_filter]
    y_df3 = y_val_df[df3_filter]
    y_df4 = y_val_df[df4_filter]

    df1_filter_desc = [(root[0], '<=', root[1]), (left[0], '<=', left[1])]
    df2_filter_desc = [(root[0], '<=', root[1]), (left[0], '>', left[1])]
    df3_filter_desc = [(root[0], '>', root[1]), (right[0], '<=', right[1])]
    df4_filter_desc = [(root[0], '>', root[1]), (right[0], '>', right[1])]
    
    print("Slices of tree filter descriptions:")
    print(df1_filter_desc)
    print(df2_filter_desc)
    print(df3_filter_desc)
    print(df4_filter_desc)

    return [((x_df1, y_df1), df1_filter_desc),
            ((x_df2, y_df2), df2_filter_desc),
            ((x_df3, y_df3), df3_filter_desc),
            ((x_df4, y_df4), df4_filter_desc)]


def get_acc_of_df(x_val_df, y_val_df, xgbModel, metric_to_use):
    y_predict = xgbModel.predict(x_val_df)
    val_acc = metric_to_use(y_val_df, y_predict)

    return val_acc


def get_acc_of_each_slice(slices_dataframes, xgbModel, metric_to_use):
    slices_with_acc = []
    for i in range(len(slices_dataframes)):
        acc_of_curr_slice = get_acc_of_df(slices_dataframes[i][0][0], slices_dataframes[i][0][1], xgbModel, metric_to_use)
        slices_with_acc.append((slices_dataframes[i], acc_of_curr_slice))
        print(f"Accuracy of slice #{i+1} is: {acc_of_curr_slice}")
    return slices_with_acc


def get_most_problematic_slice(slices_with_accuracy, min_size_threshold, accuracy_threshold):
    worst_slice_score = 0
    worst_slice_index = -1
    print(f"Min size threshold: {min_size_threshold}")

    for i in range(len(slices_with_accuracy)):
        current_slice = slices_with_accuracy[i]
        accuracy_of_slice = current_slice[1]
        size_of_slice = len(current_slice[0][0][0])
        print(f"Slice #{i+1}, Accuracy of {accuracy_of_slice} and size of {size_of_slice}")
        if size_of_slice < min_size_threshold:
            continue

        slice_score = 1 / (accuracy_of_slice)

        if slice_score > worst_slice_score:
            worst_slice_index = i
            worst_slice_score = slice_score
    print(f"Most problematic slice index: {worst_slice_index+1}")
    return (worst_slice_index != -1, slices_with_accuracy[worst_slice_index])


def get_train_df_of_problematic_slice(x_train_raw_df, x_train_df, y_train_df, problematic_slice):
    filter1_col_name = problematic_slice[0][1][0][0]
    filter1_col_val = problematic_slice[0][1][0][2]
    filter1_operation = problematic_slice[0][1][0][1]

    filter2_col_name = problematic_slice[0][1][1][0]
    filter2_col_val = problematic_slice[0][1][1][2]
    filter2_operation = problematic_slice[0][1][1][1]

    print(f"Filter #1: {filter1_col_name} {filter1_operation} {filter1_col_val}")
    print(f"Filter #2: {filter2_col_name} {filter2_operation} {filter2_col_val}")

    if filter1_operation == '<=':
        df_filter1 = (x_train_df[filter1_col_name] <= filter1_col_val)
    else:  # filter1_operation == '>'
        df_filter1 = (x_train_df[filter1_col_name] > filter1_col_val)

    if filter2_col_name != 'undefined!':
        if filter2_operation == '<=':
            df_filter2 = (x_train_df[filter2_col_name] <= filter2_col_val)
        else:  # filter1_operation == '>'
            df_filter2 = (x_train_df[filter2_col_name] > filter2_col_val)
    else:
        df_filter2 = None

    if df_filter2 is not None:
         df_filter = df_filter1 & df_filter2
    else:
        df_filter = df_filter1

    x_train_prob_slice = x_train_raw_df[df_filter]
    y_train_prob_slice = y_train_df[df_filter]

    print(f"Train problematic slice size: {len(x_train_prob_slice)}")

    return (x_train_prob_slice, y_train_prob_slice)


def filter_df_by_prob_label(x_train_prob_slice, y_train_prob_slice, problematic_label, dataset_label_column_name):
    df_filter = y_train_prob_slice[dataset_label_column_name] == problematic_label

    x_train_filtered_df = x_train_prob_slice[df_filter]
    y_train_filtered_df = y_train_prob_slice[df_filter]

    print(f"Train problematic slice size after label filer: {len(x_train_filtered_df)}")

    return (x_train_filtered_df, y_train_filtered_df)


def extract_discrete_columns_from_data(data, threshold=20):
    column_uniques = data.nunique()
    column_names = data.columns
    discrete_columns = []

    for col_index, num_of_uniques in enumerate(column_uniques):
        if num_of_uniques < threshold:  # Any column with less unique values than the trheshold is considered discrete
            discrete_columns.append(column_names[col_index])

    return discrete_columns


def generate_synthetic_data_from_slice(x_data, y_data, num_samples_to_generate, dataset_label_column_name, should_use_ctgan, num_epochs=10, discrete_columns=None):

    print("Training CTGAN on problematic slice")

    df_cp = x_data.copy(deep=True)
    y_df_cp = y_data.copy(deep=True)
    df_cp[dataset_label_column_name] = y_df_cp

    if should_use_ctgan:
        if discrete_columns is None:
            discrete_columns = extract_discrete_columns_from_data(df_cp)

        ctgan = CTGANSynthesizer(epochs=num_epochs)
        ctgan.fit(df_cp, discrete_columns)

        samples = ctgan.sample(num_samples_to_generate)
    else:
        samples = df_cp.sample(n=num_samples_to_generate)

    print(f"Generated {num_samples_to_generate} new samples for the problematic slice")

    samples_y = pd.DataFrame(samples[dataset_label_column_name])
    samples_x = samples.drop([dataset_label_column_name], axis=1)

    return (samples_x, samples_y)


def run_cycle_on_validation_dataset_of_label(X_val_of_label, 
                                             y_val_of_label, 
                                             clf, 
                                             metric_to_use, 
                                             df_len, min_support, 
                                             X_train_raw, 
                                             X_train, 
                                             y_train, 
                                             curr_label, 
                                             dataset_label_column_name, 
                                             should_use_ctgan,
                                             ctgan_num_epochs, 
                                             synthetic_samples_to_generate_percent,
                                             accuracy_threshold):

    dtc = create_tree_for_slices(X_val_of_label, y_val_of_label, clf)
    print("Created Decision Tree Classifier for finding problematic slices")

    slices_dataframes = get_dataframes_from_slices_by_tree(dtc, X_val_of_label, y_val_of_label)
    print("Extracted the slices by the tree leaves")

    slices_with_accuracy = get_acc_of_each_slice(slices_dataframes, clf, metric_to_use)
    print("Calculated accuracy of each slice")

    (found_prob_slice, problematic_slice) = get_most_problematic_slice(slices_with_accuracy, df_len * min_support, accuracy_threshold)
    print("Chose the most problematic slice")

    if found_prob_slice:
        (x_train_prob_slice, y_train_prob_slice) = get_train_df_of_problematic_slice(X_train_raw, X_train, y_train,
                                                                                        problematic_slice)
        print("Extracted the problematic slice from the train dataframe")

        (x_train_filtered_df, y_train_filtered_df) = filter_df_by_prob_label(x_train_prob_slice, y_train_prob_slice, curr_label, dataset_label_column_name)
        print("Filtered the problematic slice from the train dataframe")

        num_samples_to_generate = int(len(x_train_filtered_df) * synthetic_samples_to_generate_percent)
        
        (samples_x, samples_y) = generate_synthetic_data_from_slice(x_train_filtered_df, y_train_filtered_df, num_samples_to_generate, dataset_label_column_name, should_use_ctgan, ctgan_num_epochs)
        print("Generated new samples from the problematic slice")
    else:
        (samples_x, samples_y) = (None, None)

    ## REMOVE
    X_train_raw = X_train_raw.append(samples_x)
    y_train = y_train.append(samples_y)
    X_train = pd.get_dummies(X_train_raw, columns=cat_feats)

    # Create an XGB classifier and train it on 70% of the data set.
    clf = XGBClassifier()
    clf.fit(X_train, y_train)
    print("Fit XGBClassifier")

    slices_with_accuracy = get_acc_of_each_slice(slices_dataframes, clf, metric_to_use)

    #print(slices_with_accuracy)
    
    return (found_prob_slice, samples_x, samples_y)
        
cat_feats = None

def main_code(config_file_name):

    with open(config_file_name, 'r') as f:
        config = json.load(f)
    
    print("Loaded config.json:")
    print(config)
    
    # splitting dataset
    val_size = config["validation_size"]
    test_size = config["test_size"]
    train_size = 1 - val_size - test_size
    min_support = config["min_support"]
    iterations = config["iterations"]
    dataset_path = config["dataset_path"]
    dataset_label_column_name = config["dataset_label_column_name"]
    problematic_label = config["problematic_label"]
    synthetic_samples_to_generate_percent = config["synthetic_samples_to_generate_percent"]
    should_use_ctgan = config["should_use_ctgan"]
    ctgan_num_epochs = config["ctgan_num_epochs"]
    metric_to_use_name = config["metric_to_use_name"]
    should_generate_data_from_both_labels = config["should_generate_data_from_both_labels"]
    dataset_contains_column_names = config["dataset_contains_column_names"]
    dataset_column_names = config["dataset_column_names"]
    dataset_label_values = config["dataset_label_values"]
    dataset_label_replace_with = config["dataset_label_replace_with"]

    if metric_to_use_name == "f1":
        metric_to_use = metrics.f1_score
    elif metric_to_use_name == "recall":
        metric_to_use = metrics.recall_score
    elif metric_to_use_name == "precision":
        metric_to_use = metrics.precision_score
    else: # default to accuracy
        metric_to_use = metrics.accuracy_score

    print("Starting...")

    # Importing and displaying data
    if dataset_contains_column_names:
        data = pd.read_csv(dataset_path, delimiter=";", header='infer')
    else:
        data = pd.read_csv(dataset_path, names = dataset_column_names, delimiter=' ')
    print("Completed Dataset Loading")

    # Since y is a class variable we will have to convert it into binary format. (Since 2 unique class values)
    data[dataset_label_column_name].replace(dataset_label_values, dataset_label_replace_with, inplace=True)

    cat_feats = get_categorical_features(data, dataset_label_column_name)

    data_columns_list = pd.get_dummies(data, columns=cat_feats).columns

    # Spliting data as X -> features and y -> class variable
    data_y = pd.DataFrame(data[dataset_label_column_name])
    data_X = data.drop([dataset_label_column_name], axis=1)

    # Dividing records in training validation and testing sets along with its shape (rows, cols)
    X_train, X_test_raw, y_train, y_test = \
        train_test_split(data_X, data_y, test_size=test_size, random_state=2, stratify=data_y)

    X_train_raw, X_val_raw, y_train, y_val = \
        train_test_split(X_train, y_train, test_size=val_size / (val_size + train_size), random_state=2,
                         stratify=y_train)
    print("Split train/test/val")

    X_val = pd.get_dummies(X_val_raw, columns=cat_feats)
    X_test = pd.get_dummies(X_test_raw, columns=cat_feats)

    for i in range(len(data_columns_list)):
        if data_columns_list[i] != dataset_label_column_name and data_columns_list[i] not in X_val:
            X_val[data_columns_list[i]] = 0
        if data_columns_list[i] != dataset_label_column_name and data_columns_list[i] not in X_test:
            X_test[data_columns_list[i]] = 0

    val_label_1_filter = y_val[dataset_label_column_name] == 1
    val_label_0_filter = y_val[dataset_label_column_name] == 0

    X_val_1 = X_val[val_label_1_filter]
    X_val_0 = X_val[val_label_0_filter]
    y_val_1 = y_val[val_label_1_filter]
    y_val_0 = y_val[val_label_0_filter]

    print("Done preprocessing")

    X_train = pd.get_dummies(X_train_raw, columns=cat_feats)

    # Create an XGB classifier and train it on 70% of the data set.
    clf = XGBClassifier()
    clf.fit(X_train, y_train)
    print("Fit XGBClassifier")

    y_pred_val = clf.predict(X_val)
    y_pred_test = clf.predict(X_test)
    print(f'Regular Baseline Model Classification Report On Validation Dataset:')
    print(metrics.classification_report(y_val, y_pred_val, digits=4))

    print(f'Regular Baseline Model Classification Report On Test Dataset:')
    print(metrics.classification_report(y_test, y_pred_test, digits=4))

    first_acc = metric_to_use(y_val, y_pred_val)
    first_acc_test = metric_to_use(y_test, y_pred_test)

    best_model = clf
    best_model_acc = first_acc
    best_model_acc_test = metric_to_use(y_test, y_pred_test)

    should_run_on_label_1 = should_generate_data_from_both_labels or problematic_label == 1
    should_run_on_label_0 = should_generate_data_from_both_labels or problematic_label == 0

    accuracy_threshold = best_model_acc
    print(f"Accuracy threshold: {accuracy_threshold}")

    for i in range(iterations):
        print(f'Starting iteration #{i+1}')

        if should_run_on_label_1:
            (found_prob_slice_1, samples_x_1, samples_y_1) = run_cycle_on_validation_dataset_of_label(X_val_1, y_val_1, clf, 
                metric_to_use, len(X_val_1), min_support, X_train_raw, X_train, y_train, 1, dataset_label_column_name, 
                should_use_ctgan, ctgan_num_epochs, synthetic_samples_to_generate_percent, accuracy_threshold)

            print("Finished cycle for label 1")
        
        if should_run_on_label_0:
            (found_prob_slice_0, samples_x_0, samples_y_0) = run_cycle_on_validation_dataset_of_label(X_val_0, y_val_0, clf, 
                metric_to_use, len(X_val_0), min_support, X_train_raw, X_train, y_train, 0, 
                dataset_label_column_name, should_use_ctgan, ctgan_num_epochs, synthetic_samples_to_generate_percent, accuracy_threshold)
            print("Finished cycle for label 0")
        
        if should_run_on_label_1 and found_prob_slice_1:
            X_train_raw = X_train_raw.append(samples_x_1)
            y_train = y_train.append(samples_y_1)
            print("Added new samples to the train dataframe")
        
        if should_run_on_label_0 and found_prob_slice_0:
            X_train_raw = X_train_raw.append(samples_x_0)
            y_train = y_train.append(samples_y_0)
            print("Added new samples to the train dataframe")
        
        X_train = pd.get_dummies(X_train_raw, columns=cat_feats)

        # Create an XGB classifier and train it on 70% of the data set.
        clf = XGBClassifier()
        clf.fit(X_train, y_train)
        print("Fit XGBClassifier")

        y_pred_val = clf.predict(X_val)
        y_pred_test = clf.predict(X_test)
        print(f'Regular Baseline Model Classification Report On Validation Dataset:')
        print(metrics.classification_report(y_val, y_pred_val, digits=4))

        print(f'Regular Baseline Model Classification Report On Test Dataset:')
        print(metrics.classification_report(y_test, y_pred_test, digits=4))

        curr_acc = metric_to_use(y_val, y_pred_val)
        if curr_acc > best_model_acc:
            best_model = clf
            best_model_acc = curr_acc
            best_model_acc_test = metric_to_use(y_test, y_pred_test)

        should_run_on_label_1 = should_generate_data_from_both_labels or problematic_label == 1
        should_run_on_label_0 = should_generate_data_from_both_labels or problematic_label == 0

        accuracy_threshold = best_model_acc
        print(f"Accuracy threshold: {accuracy_threshold}")

        print("going back...")

    print("Finished all iterations, printing final models results:")

    last_acc = metric_to_use(y_val, y_pred_val)
    last_acc_test = metric_to_use(y_test, y_pred_test)

    print(f"Validation Baseline: {first_acc} VS. Validation Last: {last_acc} VS. Validation Best: {best_model_acc}")
    print(f"Test Baseline: {first_acc_test} VS. Test Last: {last_acc_test} VS. Test Best: {best_model_acc_test}")
    print(f"The new model improved baseline performance by {last_acc_test - first_acc_test} ({(last_acc_test - first_acc_test)*100}%)")
    # save best model of validation to file
    pickle.dump(best_model, open("best_model.pickle.dat", "wb"))
    print("Dumped the best model into file")

    # save last model to file
    pickle.dump(clf, open("last_model.pickle.dat", "wb"))
    print("Dumped the last model into file, exiting...")


if __name__ == "__main__":
    config_file_name = sys.argv[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main_code(config_file_name)