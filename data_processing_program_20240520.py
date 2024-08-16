def deal_special_brand(text):
    cat_check = text.split(" ")
    # Join the special brands in supplier list
    if 'XUZHOU CONSTRUCTION MACHINERY' in text:
        text = (' '.join(['XCMG', text]))
    if  'CAT' in cat_check:
        text = (' '.join(['CATERPILLAR', text]))
    if  'MANITOWOC' in text:
        text = (' '.join(['GROVE', text]))
    if  'MARUBENI' in text:
        text = (' '.join(['KOMATSU', text]))    
    if  'TOYOTA TSUSHO CORPORATION' in text:
        text = (' '.join(['TAKEUCHI', text]))
    if  'SHANDONG LINGONG CONSTRUCTION MACHINERY' in text:
        text = (' '.join(['SDLG', text]))
    if 'HİDROMEK' in text:
        text = (' '.join(['HIDROMEK', text]))
    return text


# remove the special marks
def pre_processing(description):
    description = str(description)  # Convert to str if not already
    description = deal_special_brand(description)
    description = description.replace(',', ' ')
    description = description.replace('(', ' ')
    description = description.replace(')', ' ')
    description = description.replace('[', ' ')
    description = description.replace(']', ' ')
    description = description.replace('/', '')
    description = description.replace(':', ' ')
    description = description.replace('*', '')
    description = description.replace(';', ' ')
    description = description.rstrip()
    description = description.lstrip()
    description = description.upper()
    return description


def data_preperation(df, df_ref):
    import re
    # Check the necessary col_names are contained
    required_columns = ['supplier', 'product description']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"error message: make sure dataframe contains the following column names {required_columns}.")
        
    # Create a copy of prod description to avoid changing the original data
    df['description2'] = df['product description']
    
    # uppercase the supplier col and remove extra spaces
    df['supplier'] = df.loc[:,'supplier'].apply(lambda x: pre_processing(x))
    df['supplier'] = df['supplier'].apply(lambda x: re.sub(r'\s+', ' ', x))

    df['description2'] = df.loc[:,'description2'].apply(lambda x: pre_processing(x))
    df['description2'] = df.loc[:,'description2'].apply(lambda x: re.sub(r'\s+', ' ', x))

    df_ref['model_ref'] = df_ref.loc[:,'model'].apply(lambda x: pre_processing(x))
    df_ref['brand'] = df_ref.loc[:,'brand'].apply(lambda x: re.sub(r'\s+', ' ', x))

    
def matching_program(df, df_ref, file_type):
    
    # Check if any brand existed in product description column -> Mark: Brand in description; Only take the first brand if multipul brands existed
    # Check if any brand existed in supplier description column -> Mark: Brand in supplier; Only take the first brand if multiple brands existed
    # For rows with brands -> Searching the brand in reference table -> Check if the model existed in product description -> if only a single model match -> Mark: Fully match
    # For rows with brands -> Searching the brand in reference table -> Check if the model existed in product description -> if several models match, only take the longest model -> Mark: Multiple model matched, take the longest match
    # Nothing matched -> Mark: No match
    
    # import necessary packages
    import pandas as pd
    import numpy as np

    # BLOCK 1: MATCHING REFERENCE TABLE
    error_rows_index={} # record the error index for potential debug
    for index, row in df.iterrows():
        product_description = row['description2']
        supplier = row['supplier']
        unique_brands_ref = df_ref['brand'].unique()
        brand_temp_list = []
        try:
            for brand in unique_brands_ref:
                if brand in product_description or brand in supplier: # check if the brands in the reference table exist in product description column
                    brand_temp_list.append(brand)

            # when brand exist
            if len(brand_temp_list) >= 1:
                # longest_brand = max(brand_temp_list, key=len) # take the longest string - brand
                first_brand = brand_temp_list[0] #  take the first find brand, which can be adjusted accordingly
                df.at[index, 'brand'] = first_brand # mark the brand
                filtered_reference_df = df_ref[df_ref['brand']==first_brand] # Begin by filtering the reference table by brand to minimize calculations, then search for models under that brand.
                unique_models = filtered_reference_df['model_ref'] # all models under that brand.

                model_temp_list = []
                for model in unique_models:
                    if model in product_description:
                        model_temp_list.append(model)

                if len(model_temp_list) >= 1:
                    longest_model = max(model_temp_list, key=len)
                    df.at[index,'model']= longest_model # mark model
                    df.loc[index,['capacity','type']]= filtered_reference_df.loc[filtered_reference_df['model_ref']==longest_model, ['capacity','type']].values[0] # mark working capacity and model.
                    df.at[index, 'remark'] = 'Fully match'

                else: # without a match
                    df.at[index,'model']= 'UNKNOWN' # mark as unknown
                    df.loc[index,['capacity','type']] = ['UNKNOWN','UNKNOWN'] # mark as unknown
                    df.at[index, 'remark'] = 'Brands existed but without models'

            # without a brand, and mark the rest of columns as unknown. do not apply reverse mark(define the brand by a specified model).
            else:
                df.loc[index,['brand','model','capacity','type','remark']] = ['UNKNOWN','UNKNOWN','UNKNOWN','UNKNOWN','No match']
        
        except Exception as e:
            error_rows_index[index] = str(e)
            df.at[index, 'remark'] = 'Error Occurred'

    # BLOCK 2: type, used
    # delete the irrelevant items by keywords
    irrelevant_type_list = ['CARRIER', 'TELESCOPLADER', 'HARBOUR', 'OPEN SHEET', 'STACK', 
                            'BOAT', 'BACKHOE', 'SKID', 'ROLLER', 'BENZ', 'TELEHANDLER', 
                            'LOADER', 'FORK', 'PAVER', 'STACKER', 'MATERIAL HANDLER', 
                            'BRIDGE', 'REACH', 'HANDER', 'GRABBER', 'GANTRY', 'BACK HOE', 
                            'PORT', 'MERCEDES', 'VİNCE', 'SPIDER', 'PIPE', 'HANDLING', 
                            'GLASS CRANE', 'LOAD', 'GRADER', 'GLASS CRANE', 'SPIDER']
    df = df.loc[~df['description2'].str.contains('|'.join(irrelevant_type_list), case=False, na=False)]
    
    if file_type == 'excavator':
        # search keyword 'crawler excavator' and mark the machine type
        df.loc[(df['type']=='UNKNOWN')&(df['description2'].str.contains('crawler excavator',case=False)),'type'] = 'EXCAVATOR'

        # search keyword 'wheel excavator' and mark the machine type
        df.loc[(df['type']=='UNKNOWN')&(df['description2'].str.contains('wheel excavator',case=False))&(df['description2'].str.contains('wheel',case=False)),'type'] = 'WHEEL EXCAVATOR'

        # search keyword 'AMPHIBIOUS EXCAVATOR' and mark the machine type
        df.loc[(df['type']=='UNKNOWN')&(df['description2'].str.contains('AMPHIBIOUS',case=False)),'type'] = 'AMPHIBIOUS EXCAVATOR'

        # search keyword 'tire' and mark the machine type
        df.loc[(df['description2'].str.contains('tire',case=False)),'type']='WHEEL EXCAVATOR'
        
    elif file_type == 'crane':
        df.loc[(df['description2'].str.contains('Wheel|tire',case=False))&(df['type']=='UNKNOWN'),'type'] = 'WHEELED CRANE'
        df.loc[(df['description2'].str.contains('rough',case=False))&(df['type']=='UNKNOWN'),'type'] = 'ROUGH-TERRAIN CRANE'
        df.loc[(df['description2'].str.contains('crawler',case=False))&(df['type']=='UNKNOWN'),'type'] = 'CRAWLER CRANE'
        df.loc[(df['description2'].str.contains('crawler',case=False))&(df['description2'].str.contains('telescopic',case=False)),'type'] = 'CRAWLER CRANE (TELESCOPIC_BOOM)'

    # label the used unit
    PROD_YEAR = np.arange(1950,2019,1) # Equipment between 1950-2019 is considered used
    condition_list = ['USED', 'SECOND HAND', 'SECONDHAND', 'OLD', '2ND HAND', 'REFURBISH'] # used keywords list
    condition_list += [str(year) for year in PROD_YEAR] # merge two lists

    for index, row in df.iterrows():
        description = row['description2']
        condition_temp_list = [1 if description.split(' ').count(condition_key.upper()) > 0 else 0 for condition_key in condition_list]
        condition = 'used' if sum(condition_temp_list) > 0 else 'new'
        df.at[index, 'new/used'] = condition

    # since mis-labeling could happen, double check
    df.loc[(df['description2'].str.contains('NEW|UNUSED',case=False))&(df['new/used']=='used'),'new/used']='new'

    # label out the partial shipment by [CKD, SKD and partial] keywords
    partial_list = ['CKD', 'SKD', 'partial']
    df.loc[df['description2'].str.contains('|'.join(partial_list), case=False, na=False),'remark']='Parts'
    
    return df, error_rows_index


def search_regex(df_original, condition, search_col_name, df_regex_original, filter_brand=True, mark_load=True):
    # Using regular-expression(regex) search data patterns to mark the data without labels
    import pandas as pd
    import numpy as np
    import re
    df = df_original.copy() # creat a copy
    
    for index, row in df.loc[condition].iterrows():
        description = row[search_col_name]
        brand = row['brand']
        
        if filter_brand:
            if brand != 'UNKNOWN':  # if brand has labeled but without a model
                df_regex = df_regex_original[df_regex_original['brand'] == brand]  # it will only loop the regexes under that brand
            else:
                df_regex = df_regex_original  # loop all regexes when brand is unknown
        else:
            df_regex = df_regex_original
            
        for model_regex in df_regex['model_regex']:
            model_match = re.findall(model_regex, description)
            if model_match:                   
                df.at[index, 'model'] = max(model_match, key=len) # take the longest model
                df.at[index, 'brand'] = df_regex.loc[df_regex['model_regex']==model_regex,'brand'].values[0]
                df.at[index, 'type'] = df_regex.loc[df_regex['model_regex']==model_regex,'category'].values[0]

                if len(model_match)>1: # when more than one regex are matched
                    # The followed code will keep multiple petters, but in practical serial numbers could have similar patterns that could cause a wrongly label. Here we only preserve one, and the model often appear before the serial number.
                    # df.at[index, 'model'] = ', '.join(model_match)
                    # df.at[index, 'remark'] = 'multiple matches according to the pattern'
                    if filter_brand:
                        df.at[index, 'remark'] = 'Keep the longest from the multiple matched' 
                    else:
                        df.at[index, 'remark'] = 'No brand in description, and keep the longest from the multiple matched' 
                else: # regex匹配结果唯一时
                    if filter_brand:
                        df.at[index, 'remark'] = 'Unique model match with regex' 
                    else:
                        df.at[index, 'remark'] = 'No brand in description, and unique model match with regex' 
                    
                if mark_load: # specify in the arguments that suggests mark the capacity or not
                    capacity_regex = df_regex.loc[df_regex['model_regex']==model_regex,'capacity_regex'].values[0]
                    numeric_part = re.search(capacity_regex, max(model_match, key=len))

                    if numeric_part: # if numeric part exist
                        starting_point = df_regex.loc[df_regex['model_regex']==model_regex,'starting_point'].values[0]
                        numeric_value = numeric_part.group(1)

                        if starting_point == 0: # number/10
                            capacity = float(numeric_value) / 10
                            df.at[index, 'capacity'] = capacity
                        elif starting_point == 1: # start from the second number and devide by 10
                            numeric_value = numeric_part.group(1)[1:]
                            capacity = float(numeric_value) / 10
                        elif starting_point == 2: # There is no naming rules related to the capacity, don't label
                            capacity = 'TBD' # to be defined, check individually
                        elif starting_point == 3: # mark the numerical part directly
                            capacity = float(numeric_value) 
                        elif starting_point == -2: # start from the third number and mark directly
                            numeric_value = numeric_part.group(1)[2:]
                            capacity = float(numeric_value)
                            df.at[index, 'capacity'] = capcacity
                        else: # if the mark is -1, mark the number start from the second number
                            numeric_value = numeric_part.group(1)[1:]
                            capacity = float(numeric_value)
                            df.at[index, 'capacity'] = capacity
                    else:
                        df.at[index, 'capacity'] = 'UNKNOWN'                
                        
    return df


def search_capacity(df_original, condition, search_col, capacity_regex):
    import re
    # capacity_regex = r'\b(\d+)\s*(?:METRIC\s*)?TONS?\b'
    # (?:METRIC\s*)?: Matches an optional 'METRIC' followed by optional whitespace characters. 
    # The (?: ... ) is a non-capturing group.
    df = df_original.copy()
    for index, row in df.loc[condition].iterrows():
        description = row[search_col]
        match = re.search(capacity_regex, description)
        if match:
            capacity = match.group(1)
            df.at[index, 'capacity'] = capacity
            df.at[index, 'remark'] = 'Description contains working capacity'
    return df


def mark_unknown_model_with_exsisted_lifting_capacity(df_original):
    df = df_original.copy()
    # reverse mark the models with existed information of lifting capacity, machine type, and brand.
    # for instance, if there is a record in the dataset that contains all information, and some records has similar information but incompelete, this code will inference the unknown info based on those already known.
    for index,row in df.iterrows():
        brand = row['brand']
        model = row['model']
        capacity = row['capacity']
        model_type = row['type']
        model_info = df.loc[(df['brand']==brand)&(df['model']!='UNKNOWN'),['model','capacity','type']].drop_duplicates()
        # Capacity, brand and type are known but model. Inference the model base on existed knowledge of the dataset
        if capacity!='UNKNOWN' and brand!='UNKNOWN' and model=='UNKNOWN' and model_type!='UNKNOWN':
            if model_type in model_info['type'].unique(): # if type existed
                model_info = model_info[model_info['type']==model_type]
                # threshhold within +-5%
                for exsisted_capacity in model_info['capacity']:
                    exsisted_capacity = float(exsisted_capacity)
                    capacity = float(capacity)
                    if exsisted_capacity > capacity*0.95 and exsisted_capacity < capacity*1.05:
                        df.at[index,'model'] = model_info.loc[model_info['capacity']==exsisted_capacity,'型号'].iloc[0]
                        df.at[index,'remark'] = 'Description contains working capacity, and the model is inferenced with existed infomation'
    return df


def check_parts(result):
    print(f"items with 'partial': {(result['description2'].str.contains('partial', case=False) == True).sum()}")
    print(f"items with 'party': {(result['description2'].str.contains('party', case=False) == True).sum()}")
    print(f"items with 'part': {(result['description2'].str.contains('part', case=False) == True).sum()}")
    print(f"items with 'assemble': {(result['description2'].str.contains('assemble', case=False) == True).sum()}")
    print(f"items with 'SKD': {(result['description2'].str.contains('skd', case=False) == True).sum()}")
    

def mark_outliers(df_original, term=True):
    # If the dataset specified the trading term, the term argument should fill in True
    df = df_original.copy()
    # Should calculating the midium price under same trading terms if term exist.
    # Unknown and used model should marked as 'unknown' in 'outlier' column; 
    # This only apply for the excavator: if model is known, check if the capacity is within median weight*(100% ± 20%), and the exceptions should be marked as outlier.
    # check the prices of the same model are within the range of the median price of that model*(100% ± 20%), and the exceptions should be marked as outlier.
    if term: 
        # check if the dataset has the col
        required_columns = ['term']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"error message: make sure dataframe contains the following column names {required_columns}.")
                
        midian_price = df.groupby(['brand','model','term'])['price in usd'].agg('median').reset_index()
        df['outliers'] = ["UNKNOWN" 
         if model_label == 'UNKNOWN' or load == 'UNKNOWN' or used_new == 'used'
         else "yes" if unit_price >= 1.2*(midian_price[(midian_price['brand']==brand) & (midian_price['model']==model_label) & (midian_price['term']==term)]['price in usd'].values[0])
         else "yes" if unit_price <= 0.8*(midian_price[(midian_price['brand']==brand) & (midian_price['model']==model_label)]['price in usd'].values[0])
         else "no" 
         for brand, model_label, load, unit_price, used_new, term in zip(df['brand'], df['model'], df['capacity'], df['price in usd'], df['new/used'], df['term'])]
    else: # without term
        midian_price = df.groupby(['brand', 'model'])['price in usd'].agg('median').reset_index()
        df['outliers'] = ["UNKNOWN" 
         if model_label == 'UNKNOWN' or load == 'UNKNOWN' or used_new == 'used'
         else "yes" if unit_price >= 1.2*(midian_price[(midian_price['brand']==brand) & (midian_price['model']==model_label)]['price in usd'].values[0])
         else "yes" if unit_price <= 0.8*(midian_price[(midian_price['brand']==brand) & (midian_price['model']==model_label)]['price in usd'].values[0])
         else "no" 
         for brand, model_label, load, unit_price, used_new in zip(df['brand'], df['model'], df['capacity'], df['price in usd'], df['new/used'])]
    
    return df


# 计算人民币汇率单价
def convert_usd_to_cny(df_original, rate_dict): 
    # The rate should be assigned outside the function
    # rate_dict = {2023: {1: 6.7604, 2: 6.9519, 3: 6.8717, 4: 6.924, 5: 7.0821, 6: 7.2258, 7: 7.1305, 8: 7.1811, 9: 7.1798, 10: 7.1779, 11: 7.1018, 12: 7.0827},
             # 2024: {1: 7.1039, 2: 7.1036, 3: 7.0950, 4:7.1063, 5:7.1088}}
    import pandas as pd    
    df = df_original.copy()
    df['date'] = pd.to_datetime(df['date'])
    for year, month_dict in rate_dict.items():
        for month, rate in month_dict.items():
            # Filter DataFrame for the current year and month
            filtered_df = df[(df['date'].dt.year == year) & (df['date'].dt.month == month)]

            # Check if the necessary columns exist
            required_columns = ['date', 'price in usd', 'amount in usd']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"The DataFrame is missing one or more required columns from {required_columns}.")

            # Convert USD price and amount to RMB
            df.loc[(df['date'].dt.year == year) & (df['date'].dt.month == month), 'price in cny'] = filtered_df['price in usd'] * rate
            df.loc[(df['date'].dt.year == year) & (df['date'].dt.month == month), 'amount in cny'] = filtered_df['amount in usd'] * rate
    return df


def define_load_interval(df, file_type = 'excavator', load_interval = 50):
    import pandas as pd
    # convert the capcatify and weight to numeric values, and unknown strings here will be transfered to na values.
    # check the required cols
    required_columns = ['unit weight in ton']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"make sure the following column names are existed {required_columns}. Otherwise, please create the col with formular: weight in kg/qty/1000")
                
    df['capacity'] = pd.to_numeric(df['capacity'], errors='coerce')
    df['unit weight in ton'] = pd.to_numeric(df['unit weight in ton'], errors='coerce')
    
    # Use the max value between weight and capacity as the upper threshold
    if file_type == 'excavator':
        max_load = max(df.loc[pd.notna(df['capacity']), 'capacity'].max(),df.loc[df['unit weight in ton']!=0, 'unit weight in ton'].max())
    if file_type == 'crane':
        max_load = df.loc[pd.notna(df['capacity']), 'capacity'].max()
        
    upper_limit_index = int(round(max_load/load_interval, 0) + 1)
    
    if file_type == 'excavator':
        for index, row in df.iterrows():
            load = row['capacity']
            weight = row['unit weight in ton']

            if pd.notna(load) and load!=0:
                evaluate_value = load
            else:
                evaluate_value = weight
                
            if evaluate_value == 0:
                df.at[index, 'capacity interval'] = 'UNKNOWN'
            elif evaluate_value < 5:
                df.at[index, 'capacity interval'] = '<5T'
            elif evaluate_value>=5 and evaluate_value<10:
                df.at[index, 'capacity interval'] = '5-10T'
            else:
                for i in range(1, upper_limit_index):
                    lower_bound = i * load_interval
                    upper_bound = (i + 1) * load_interval
                    if evaluate_value >= lower_bound and evaluate_value < upper_bound:
                        df.at[index, 'capacity interval'] = f'{lower_bound}-{upper_bound}T'
                    i = i+1
            
    if file_type == 'crane':
        for index, row in df.loc[pd.notna(df['capacity'])].iterrows():
            load = row['capacity']

            if load == 0:
                df.at[index, 'capacity interval'] = 'UNKNOWN'
            elif load < load_interval:
                df.at[index, 'capacity interval'] = f'<{load_interval}T'
            else:
                for i in range(1, upper_limit_index + 1):
                    lower_bound = i * load_interval
                    upper_bound = (i + 1) * load_interval
                    if load >= lower_bound and load < upper_bound:
                        df.at[index, 'capacity interval'] = f'{lower_bound}-{upper_bound}T'
                    i = i + 1

    df['capacity'] = df['capacity'].fillna('UNKNOWN')



def define_excavator_load_type_interval(df):
    # according to the working scenarios define the interval, and those over 70 tons are defined as mining excavators
    import pandas as pd
    for index, row in df.iterrows():
        load = row['capacity']
        weight = row['unit weight in ton']
        
        if load != 'UNKNOWN' and load!=0:
            evaluate_value = load
        else:
            evaluate_value = weight
            
        if evaluate_value == 0 or pd.isna(evaluate_value):
            df.loc[index, 'type interval'] = 'UNKNOWN'
        elif evaluate_value < 5:
            df.loc[index, 'type interval'] = '<5T'
        elif evaluate_value>=5 and evaluate_value<10:
            df.loc[index, 'type interval'] = '5-10T'
        elif evaluate_value>=10 and evaluate_value<30:
            df.loc[index, 'type interval'] = '10-30T'
        elif evaluate_value>=30 and evaluate_value<70:
            df.loc[index, 'type interval'] = '30-70T'
        elif evaluate_value>=70 and evaluate_value<90:
            df.loc[index, 'type interval'] = '70-90T'
        else:
            df.loc[index, 'type interval'] = '≥90T'
    
    
    
def update_regex_df(df_regex, update_list):
    # update the regex tables
    import pandas as pd
    # update format should be as the follows
    # update_list = [{'brand': 'KOMATSU', 'model_regex': r'HB\d{3}-\d', 'capacity_regex':r'HB(\d+)', 'category':'EXCAVATOR', 'starting_point':0}]
    update_df = pd.DataFrame(update_list)
    df_regex = pd.concat([df_regex,update_df],axis=0,ignore_index=True)
    # drop duplicates
    df_regex = df_regex.drop_duplicates(subset=['brand','model_regex'])
    # sort the regex by string length
    df_regex.sort_values(by='model_regex', key=lambda x: x.str.len(), ascending=False, inplace=True)
    df_regex = df_regex.reset_index(drop=True)
    return df_regex


def check_col_names(df_col_names, ref_col_names):
    # Col names consistancy check
    # Convert the column names of both DataFrames to sets
    columns_set1 = set(df_col_names)
    columns_set2 = set(ref_col_names)

    # Find the columns that are in df1 but not in df2
    columns_only_in_df1 = columns_set1 - columns_set2

    # Find the columns that are in df2 but not in df1
    columns_only_in_df2 = columns_set2 - columns_set1

    # Check if there are any differences
    if columns_only_in_df1:
        print(f"Columns in the original df but not in the reference: {columns_only_in_df1}")
    if columns_only_in_df2:
        print(f"Columns in the reference df but not in original: {columns_only_in_df2}")
    if columns_set1 == columns_set2:
        print('Consistent listing')
        
        
def get_key_from_value(dictionary, target_value):
    for key, value in dictionary.items():
        if value == target_value:
            return key
    # Return None if the value is not found
    return None


## Mark zoomlion product        
# zoomlion_dict = {'ZE60E-10':5950, 'ZE75E-10':7500, 'ZE35GU':3790, 'ZE18GU':1800, 'ZE135E-10':14000, 
#                  'ZE215E':21500, 'ZE335E':33100, 'ZE370E': 36000, 'ZE490EK-10':49500}
# from data_processing_program_20240112 import get_key_from_value

# for index, row in df2.loc[df2['remark'].isin(['No match','Brands existed but without models'])&(df2['brand']=='zoomlion'.upper())].iterrows():
#     weight = row['weight in kg']
#     for load in zoomlion_dict.values():
#         if weight >= load*0.95 and weight < load*1.05:
#             zoomlion_model = get_key_from_value(zoomlion_dict, load)
#             df2.loc[index, ['model','capacity','type','remark']] = [zoomlion_model, load/1000, 'excavator'.upper(),'inferenced model base on its weight']
#         else:
#             pass


def matching_program_individual(df_original, condition, df_ref):
    df = df_original.copy()
    error_rows_index={} # mark out the bug indexes for further debug
    for index, row in df[condition].iterrows():
        brand = row['brand']
        product_description = row['description2']
        filtered_reference_df = df_ref[df_ref['brand']==brand]
        refrence_model_list = df_ref.loc[df_ref['brand']==brand,'model_ref'].values
        try:         
            # there are models of that brand in the reference table
            if len(refrence_model_list) > 0:
                
                model_temp_list = []
                for model in refrence_model_list:
                    if model in product_description:
                        model_temp_list.append(model)
                        
                if len(model_temp_list) >= 1:
                    longest_model = max(model_temp_list, key=len)
                    df.at[index,'model']= longest_model # mark model
                    df.loc[index,['capacity','hp','type']]= filtered_reference_df.loc[filtered_reference_df['model_ref']==longest_model, ['capacity','hp','type']].values[0] # mark the capcity and model
                    df.at[index, 'remark'] = 'Fully match'

                else: # if no match
                    df.at[index,'model']= 'UNKNOWN'
                    df.loc[index,['capacity','type']] = ['UNKNOWN','UNKNOWN']
                    df.at[index, 'remark'] = 'Brands existed but without models'

            # No information match
            else:
                df.at[index,'remark'] = 'No match'
        
        except Exception as e:
            error_rows_index[index] = str(e)
            df.at[index, 'remark'] = 'Error Occurred'
            
    return df, error_rows_index


def new_or_used(df,search_col):
    # Mark used machines
    PROD_YEAR = np.arange(1950,2019,1) # in between 1950-2019 are counted as used machine
    condition_list = ['USED', 'SECOND HAND', 'SECONDHAND', 'OLD', '2ND HAND', 'REFURBISH'] # keywords of used machines
    condition_list += [str(year) for year in PROD_YEAR]

    for index, row in df.iterrows():
        description = row[search_col]
        condition_temp_list = [1 if description.split(' ').count(condition_key.upper()) > 0 else 0 for condition_key in condition_list]
        condition = 'used' if sum(condition_temp_list) > 0 else 'new'
        df.at[index, 'new/used'] = condition

    # double check the wrongly marked items
    df.loc[(df[search_col].str.contains('NEW|UNUSED',case=False))&(df['new/used']=='used'),'new/used']='new'
    
    
    
unit_regex_eng = r'(\d+)\s*UNITS'

def extract_units(df, search_col, keywords, regex):
    for index,row in df[df[search_col].str.contains(keywords,case=False)].iterrows():
        description = row[search_col]
        match = re.search(regex, description)
        if match:
            qty = int(match.group(1))
            df.loc[index,['qty','remark']]= [qty,'Description contains quantity keywords']
                
                

def unify_qty_weight(df):
    df['qty'] = pd.to_numeric(df['qty'], errors='coerce')
    df['weight in kg'] = pd.to_numeric(df['weight in kg'], errors='coerce')

    # calculate the unit price and weight
    df['price in usd'] = df['amount in usd']/df['qty']
    df['unit weight in ton'] = df['weight in kg']/df['qty']/1000
    
    

def remove_price_outliers(df,amount):    
    # remove the items that amount under 10000 usd
    df = df[~(df['amount in usd']<amount)]

    # remove the items that price under 10000 usd
    df = df[df['price in usd']>=amount]
    df = df.reset_index(drop=True)
    
    

def extract_number_word(df_orignal, search_col):
    import re
    df = df_orignal.copy()
    number_words_to_digits = {
        'ONE': 1,
        'TWO': 2,
        'THREE': 3,
        'FOUR': 4,
        'FIVE': 5,
        'SIX': 6,
        'SEVEN': 7,
        'EIGHT': 8,
        'NINE': 9,
        'TEN': 10
    }
        
    for index, row in df.iterrows():
        description = row['product description']
    
        match = re.search(r'\b(ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN)\b', description, re.IGNORECASE)
        if match:
            number_word = match.group(1).upper()
            qty = number_words_to_digits[number_word]
            df.loc[index, ['qty','remark']] = [qty, 'Description contains quantity keywords']
            
    return df


def key_players_table(df):
    # Step 1: Group by brand and calculate the sums
    grouped_df = df.groupby('brand').agg({'amount in usd': 'sum','qty': 'sum'}).reset_index()

    # Step 2: Calculate the total amounts for proportion calculations
    total_amount_usd = grouped_df['amount in usd'].sum()
    total_qty = grouped_df['qty'].sum()

    # Step 3: Calculate the proportions
    grouped_df['amount proportion'] = grouped_df['amount in usd'] / total_amount_usd * 100
    grouped_df['unit proportion'] = grouped_df['qty'] / total_qty * 100

    grouped_df = grouped_df.sort_values(by='amount proportion')

    # Step 4: Calculate the cumulative percentage
    grouped_df['cumulative proportion'] = grouped_df['amount proportion'].cumsum()

    grouped_df['brand2'] = grouped_df['brand']
    grouped_df.loc[grouped_df['cumulative proportion']<20, 'brand2'] = 'OTHERS'

    grouped_df = grouped_df.groupby(['brand2'])['amount in usd', 'qty', 'amount proportion', 'unit proportion'].sum().reset_index().sort_values(by='amount proportion',ascending=False)

    # Format the proportions as percentages
    grouped_df['amount proportion'] = grouped_df['amount proportion'].map('{:.1f}%'.format)
    grouped_df['unit proportion'] = grouped_df['unit proportion'].map('{:.1f}%'.format)
    grouped_df['amount in usd'] = (grouped_df['amount in usd'] / 1_000_000).map('{:.1f}'.format).astype(str) + ' M'
    grouped_df.columns = ['Brand', 'Amount $', 'Units #', 'Amount %', 'Unit %']
    
    row_of_others = grouped_df[grouped_df['Brand']=='OTHERS']
    row_of_unknown = grouped_df[grouped_df['Brand']=='UNKNOWN']
    df_dropped = grouped_df[~grouped_df['Brand'].isin(['OTHERS','UNKNOWN'])]
    
    df_reordered = df_dropped.append(row_of_others, ignore_index=True)
    df_reordered = df_reordered.append(row_of_unknown, ignore_index=True)
    
    return df_reordered.reset_index(drop=True)



def show_key_players(df_keyplayer):
    import matplotlib.pyplot as plt
    df_keyplayer_plot = df_keyplayer.copy()
    df_keyplayer_plot['Amount $'] = df_keyplayer_plot['Amount $'].apply(lambda x: x.rstrip(' M'))
    df_keyplayer_plot['Amount $'] = pd.to_numeric(df_keyplayer_plot['Amount $'], errors='coerce')

    colors = ['#EAE6E1','#C9C9C9','#F0EAD6','#B0C4DE','#B7B76B','#FFDAB9','#FFC0CB','#E6E6FA','#D8C9A4','#D8C9A8']
    
    plt.figure(figsize=(4, 4),dpi=120)
    plt.pie(df_keyplayer_plot['Amount $'], labels=df_keyplayer_plot['Brand'], autopct='%1.1f%%', startangle=140,colors=colors)
    plt.title('Key players')
    plt.show()
    
    
    
def key_capacity_interval(df):
    # Step 1: Group by brand and calculate the sums
    grouped_df = df.groupby('capacity interval').agg({'amount in usd': 'sum','qty': 'sum'}).reset_index()

    # Step 2: Calculate the total amounts for proportion calculations
    total_amount_usd = grouped_df['amount in usd'].sum()
    total_qty = grouped_df['qty'].sum()

    # Step 3: Calculate the proportions
    grouped_df['amount proportion'] = grouped_df['amount in usd'] / total_amount_usd * 100
    grouped_df['unit proportion'] = grouped_df['qty'] / total_qty * 100

    grouped_df = grouped_df.groupby(['capacity interval'])['amount in usd', 'qty', 'amount proportion', 'unit proportion'].sum().reset_index().sort_values(by='amount proportion',ascending=False)

    # Format the proportions as percentages
    grouped_df['amount proportion'] = grouped_df['amount proportion'].map('{:.1f}%'.format)
    grouped_df['unit proportion'] = grouped_df['unit proportion'].map('{:.1f}%'.format)
    grouped_df['amount in usd'] = (grouped_df['amount in usd'] / 1_000_000).map('{:.1f}'.format).astype(str) + ' M'
    grouped_df.columns = ['Capacity', 'Amount $', 'Units #', 'Amount %', 'Unit %']
    
    row_of_unknown = grouped_df[grouped_df['Capacity']=='UNKNOWN']
    df_dropped = grouped_df[~grouped_df['Capacity'].isin(['UNKNOWN'])]
    
    df_reordered = df_dropped.append(row_of_unknown, ignore_index=True)
    
    return df_reordered.reset_index(drop=True)



def top3_players(df):
    return key_players_table(df)[0:3]