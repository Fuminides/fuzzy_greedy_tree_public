import os
import re

import pandas as pd
import numpy as np


def read_keel_header(file_path):
        '''
        Reads the header of the provided dataset file in KEEL format.
        The function stores the dataset parameters in the "header" attribute and the
        position of the first line after the header in the "data_startline" attribute.
        :param file_path: Path to the dataset file.
        :type file_path: str
        '''
        attributes = {}
        header = {
            'relation': {},
            'inputs': {},
            'outputs': {}
        }
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if line.endswith('\n'):
                    line = line[:-1]
                parts = line.split(' ')
                # Remove possible blank spaces at the end of the line
                parts = list(filter(lambda x: x != '', parts))  

                if parts[0] == '@relation':
                    header['relation'] = parts[1]
                elif parts[0] == '@attribute' or parts[0] == '@Attribute':
                    attr_name = parts[1]
                    # CATEGORICAL VALUES:
                    if '{' in attr_name:
                        # Categorical data
                        attr_name, values = attr_name.split('{')
                        attributes[attr_name] = {}
                        attributes[attr_name]['type'] = 'categorical'
                        values[:-1].strip()  # Remove all possible blank spaces
                        range_vals = values[:-1].split(',')
                        attributes[attr_name]['range'] = range_vals
                    elif len(parts) == 3 and '{' in parts[2]:
                        # If there's an space between the attribute name and its values:
                        attr_name = attr_name
                        values = parts[2][1:-1]
                        attributes[attr_name] = {}
                        attributes[attr_name]['type'] = 'categorical'
                        values = values.split(',')
                        range_vals = values
                        attributes[attr_name]['range'] = range_vals
                    elif len(parts) > 3 and '{' in parts[2]:
                        # If there's an space between the attribute name and its values:
                        attr_name = attr_name
                        values = parts[2:]
                        attributes[attr_name] = {}
                        attributes[attr_name]['type'] = 'categorical'
                        values[0] = values[0].strip('{')
                        values[-1] = values[-1].strip('}')
                        for i in range(len(values)):
                            values[i] = values[i].split(',')[0]
                        range_vals = values
                        attributes[attr_name]['range'] = range_vals
                    # REAL VALUES:
                    elif parts[2].startswith('real'):
                        attributes[attr_name] = {}
                        attributes[attr_name]['type'] = 'real'
                        if len(parts) == 5:
                            # Sometimes there is a blank space between real and the attributes, and others there's not
                            range_vals = [*re.findall('\d+\.\d+', parts[3]), *re.findall('\d+\.\d+', parts[4])]
                        else:
                            range_vals = re.findall('\d+\.\d+', parts[2])
                        if range_vals == []:
                            range_vals = [-np.inf, np.inf]
                        attributes[attr_name]['range'] = (float(range_vals[0]), float(range_vals[1]))
                    # INTEGER VALUES:
                    elif parts[2].startswith('integer'):
                        attributes[attr_name] = {}
                        attributes[attr_name]['type'] = 'integer'
                        if len(parts) == 5:
                            range_vals = [*re.findall(r'\d+', parts[3]), *re.findall('\d+', parts[4])]
                        elif len(parts) == 4:
                            range_vals = re.findall(r'\d+', parts[3])
                        else:
                            range_vals = re.findall(r'\d+', parts[2])
                        attributes[attr_name]['range'] = (int(range_vals[0]), int(range_vals[1]))
                elif parts[0] == '@inputs':
                    for input in parts[1:]:
                        if input.endswith(','):
                            input = input[:-1]
                        if input != '':
                            # Filter possible final blank spaces
                            header['inputs'][input] = attributes[input]
                elif parts[0] == '@outputs' or parts[0] == '@output':
                    for output in parts[1:]:
                        header['outputs'][output] = attributes[output]
                elif parts[0] == '@data':
                    return header, i
                
        return header, -1


def read_data(file_path, class_names=None, random_state=33):
    '''
    Reads and returns the data of the required partition/s of the provided dataset file in KEEL format.
    
    :param file_path: Path to the dataset file.
    :type file_path: str
    :class_names: Names of the classes of the dataset. If None, the class names will be inferred from the dataset.
    :type class_names: list<str>
    :param train: If None, all samples in the dataset will be returned. 
        If True, the first 80% examples are returned (train-val split). Otherwise, the last 20% examples are returned (test split).
    :type train: bool
    :param precomputed_partitions: If True, precomputed (randomly and stratified) files dataset_train.dat and dataset_test.dat will be used.
        Otherwise, partitions will be computed from the same file as described by param "train".
    :type precomputed_partitions: bool
    '''
    header, data_startline = read_keel_header(file_path)
    if class_names is None:
        # If class associations haven't been provided, generate them (to rename them to values from 0 to num_classes-1):
        # Create association between dataset classes and predicted classes:
        class_names_ = {}
        output_attr_name = next(iter(header['outputs']))
        for i, class_name in enumerate(header['outputs'][output_attr_name]['range']):
            class_names_[class_name] = i
            # Read samples from the dataset:
    else:
        class_names_ = class_names
    # Read the corresponding samples:
    data = pd.read_csv(file_path, sep=',', header=data_startline,
                        names=(*header['inputs'].keys(), *header['outputs'].keys()), skipinitialspace=True)
    # Remove possible blank spaces at the end of the line
    for column in data.columns:
        if data[column].dtype == 'object':  # Check if the column is of type object (string)
            data[column] = data[column].map(lambda x: x.strip() if isinstance(x, str) else x)

    data_input = data[[key for key in header['inputs'].keys()]]
    # One-hot encode categorical input variables:
    categorical_columns = [key for key in header['inputs'].keys() if header['inputs'][key]['type'] == 'categorical']
    boolean_cateogrical_columns = [key in categorical_columns for key in header['inputs'].keys() ]
    boolean_real_columns = np.logical_not(boolean_cateogrical_columns)

    # There may be categorical variables with values that are not present in the training set:
    # In order to ensure that pd.get_dummies creates a column for such non-existent values, we set the type of each categorical column
    # to pd.CategoricalDtype prior to calling pd.get_dummies, specifying all potential values (even non-present ones): 
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy)
    column_types = {}
    for column in categorical_columns:
        comun_data, unique_classes = pd.factorize(data_input[column])
        data_input = data_input.assign(**{column: comun_data})
    #data_input = data_input.astype(column_types)
    # data_input = pd.get_dummies(data_input, columns=categorical_columns)
    
    data_target = data[header['outputs'].keys()]
    # Rename classes to the ordered keys:
    output_name = list(header['outputs'].keys())[0]
    data_target = data_target[output_name].apply(lambda x: class_names_[str(x)])
    # Standardize all columns:
    # data_input = (data_input - data_input.mean()) / (data_input.std() + float_info.epsilon)
    '''if standarize:
        data_input = data_input.copy()
        data_input.loc[:, boolean_real_columns] = data_input.loc[:, boolean_real_columns].astype(float)
        data_different_min = data_input.loc[:, boolean_real_columns].min()
        data_different_max = data_input.loc[:, boolean_real_columns].max()
        import warnings
        warnings.filterwarnings("ignore")
        data_input.loc[:, boolean_real_columns] = (data_input.loc[:, boolean_real_columns] - data_different_min) / (
            data_different_max - data_different_min)'''
    
    # Code the categorical variables as integers:
    '''from sklearn import preprocessing
    for column in categorical_columns:
        le = preprocessing.LabelEncoder()
        aux = le.fit_transform(data_input.loc[:, column])
        data_input.drop(column, axis=1, inplace=True)
        # Change the data type of the column to the data type of aux
        data_input[column] = aux'''
        

    return data_input, data_target, header, class_names_, boolean_cateogrical_columns
        

def load_KeelDataset_aux(file_path, batch_size, train_proportion=0.8, random_state=33):
    '''
    :param file_path: Path to the dataset file.
    :type file_path: str
    :param batch_size: Number of samples in each minibatch.
    :type batch_size: int
    :param train: If None, all samples in the dataset will be returned. 
        If True, the first 80% examples are returned (train-val split). Otherwise, the last 20% examples are returned (test split).
    :type train: bool
    :param train_proportion: Proportion of the dataset that will be used for training. The rest will be used for validation.
    :type train_proportion: float
    :param precomputed_partitions: If True, precomputed (randomly and stratified) files dataset_train.dat and dataset_test.dat will be used.
        Otherwise, partitions will be computed from the same file as described by param "train".
    :type precomputed_partitions: bool
    :param num_workers: Number of workers used to load the data.
    :type num_workers: int
    :param pin_memory: If True, the data will be stored in the device/CUDA memory before returning them.
    :type pin_memory: bool
    '''
    # 1.-Prepare the datasets:
    
    X, y, header, class_names, boolean_categorical_vector = read_data(file_path)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_proportion, random_state=random_state, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=random_state, stratify=y_train)
        
    return X_train, X_val, X_test, y_train, y_val, y_test, header, boolean_categorical_vector


def load_keel_dataset(file_path, batch_size, train_proportion=0.8, random_seed=33):
    '''
    :param file_path: Path to the dataset file.
    :type file_path: str
    :param batch_size: Number of samples in each minibatch.
    :type batch_size: int
    :param train: If None, all samples in the dataset will be returned. 
        If True, the first 80% examples are returned (train-val split). Otherwise, the last 20% examples are returned (test split).
    :type train: bool
    :param train_proportion: Proportion of the dataset that will be used for training. The rest will be used for validation.
    :type train_proportion: float
    :param precomputed_partitions: If True, precomputed (randomly and stratified) files dataset_train.dat and dataset_test.dat will be used.
        Otherwise, partitions will be computed from the same file as described by param "train".
    :type precomputed_partitions: bool
    :param num_workers: Number of workers used to load the data.
    :type num_workers: int
    :param pin_memory: If True, the data will be stored in the device/CUDA memory before returning them.
    :type pin_memory: bool
    '''
    X_train, X_val, X_test, y_train, y_val, y_test, header, boolean_categorical_vector = load_KeelDataset_aux(file_path, batch_size, train_proportion, random_state=random_seed)
    columns = list(X_train.columns)
    # Normalize scaling the data:
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    # Scale only the non-categorical columns:
    boolean_categorical_vector = np.array(boolean_categorical_vector)
    X_train = X_train.values
    X_val = X_val.values
    X_test = X_test.values

    if sum(~boolean_categorical_vector) > 0:
        X_train[:, ~boolean_categorical_vector] = scaler.fit_transform(X_train[:, ~boolean_categorical_vector])
        X_val[:, ~boolean_categorical_vector] = scaler.transform(X_val[:, ~boolean_categorical_vector])
        X_test[:, ~boolean_categorical_vector] = scaler.transform(X_test[:, ~boolean_categorical_vector])

    
    
    return X_train, X_val, X_test, y_train, y_val, y_test, header, boolean_categorical_vector


if __name__ == '__main__':
    dataset_name = 'iris'
    file_path = os.path.join('..\\keel_datasets-master', dataset_name, dataset_name + '.dat')
    #trainDataset = KeelDatasetPytorch(file_path=file_path, train=True)
    #print('Number of samples in the dataset: {}'.format(len(trainDataset)))
    #sample = trainDataset[0]
    #print('Input example: {}, {}'.format(sample[0], sample[1]))
    #print()

    X_train, X_val, X_test, y_train, y_val, y_test, header, boolean_categorical_vector = load_keel_dataset(file_path, batch_size=32, train_proportion=0.8, random_seed=33)

    # Train a logistic regression model:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    
    model = LogisticRegression(max_iter=1000, random_state=33)
    model.fit(X_train, y_train)
    
    print('Finished training!')
    
    # Test the model:
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy of the model: {}'.format(accuracy))

    