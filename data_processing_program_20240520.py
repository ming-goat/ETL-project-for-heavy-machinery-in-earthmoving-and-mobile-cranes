def deal_special_brand(text):
    cat_check = text.split(" ")
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
    
    # 1. 对照表里的品牌是否存在于产品描述里 -> 存在标记品牌；备注：描述中存在品牌；存在多个匹配品牌时，取最长的品牌
    # 2. 对照表里的品牌是否存在于供应商描述里 -> 存在标记品牌；备注：供应商存在品牌；存在多个匹配品牌时，取最长的品牌
    # 3. 存在品牌的条目 -> 对照表筛选该品牌 -> 标记匹配的型号；型号唯一时；备注：完美匹配
    # 4. 存在品牌的条目 -> 对照表筛选该品牌 -> 标记匹配的型号；有多个型号匹配时，取最长的型号；备注：多个型号匹配，取最长的型号
    # 5. 没有匹配的品牌 -> 通过regex对条目进行搜索
    # 6. 什么都没有的时候 -> 备注：无品牌型号匹配
    
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
                if brand in product_description or brand in supplier: # 判断对照表里的品牌是否存在于产品描述里
                    brand_temp_list.append(brand)

            # 存在品牌的情况下
            if len(brand_temp_list) >= 1:
                # longest_brand = max(brand_temp_list, key=len) # 取最长的品牌
                first_brand = brand_temp_list[0] # 取第一个品牌，可以根据实际调整
                df.at[index, 'brand'] = first_brand # 标记品牌
                filtered_reference_df = df_ref[df_ref['brand']==first_brand] # 根据匹配的型号筛选对照表，仅对筛选后的型号进行搜索，为型号匹配准备      
                unique_models = filtered_reference_df['model_ref'] # 筛选后品牌下的所有对照型号

                model_temp_list = []
                for model in unique_models:
                    if model in product_description:
                        model_temp_list.append(model)

                if len(model_temp_list) >= 1:
                    longest_model = max(model_temp_list, key=len)
                    df.at[index,'model']= longest_model #标记型号
                    df.loc[index,['capacity','type']]= filtered_reference_df.loc[filtered_reference_df['model_ref']==longest_model, ['capacity','type']].values[0] # 标记吨位和型号
                    df.at[index, 'remark'] = 'Fully match'

                else: # 如果没有匹配的型号
                    df.at[index,'model']= 'UNKNOWN' #标记未知
                    df.loc[index,['capacity','type']] = ['UNKNOWN','UNKNOWN'] # 标记吨位和型号未知
                    df.at[index, 'remark'] = 'Brands existed but without models'

            # 不存在品牌的情况下，直接标记，不反向搜索（通过型号判断品牌）
            else:
                df.loc[index,['brand','model','capacity','type','remark']] = ['UNKNOWN','UNKNOWN','UNKNOWN','UNKNOWN','No match']
        
        except Exception as e:
            error_rows_index[index] = str(e)
            df.at[index, 'remark'] = 'Error Occurred'

    # BLOCK 2: 类型、新旧标记
    # 删除包含非相关关键字的条目
    irrelevant_type_list = ['CARRIER', 'TELESCOPLADER', 'HARBOUR', 'OPEN SHEET', 'STACK', 
                            'BOAT', 'BACKHOE', 'SKID', 'ROLLER', 'BENZ', 'TELEHANDLER', 
                            'LOADER', 'FORK', 'PAVER', 'STACKER', 'MATERIAL HANDLER', 
                            'BRIDGE', 'REACH', 'HANDER', 'GRABBER', 'GANTRY', 'BACK HOE', 
                            'PORT', 'MERCEDES', 'VİNCE', 'SPIDER', 'PIPE', 'HANDLING', 
                            'GLASS CRANE', 'LOAD', 'GRADER', 'GLASS CRANE', 'SPIDER']
    df = df.loc[~df['description2'].str.contains('|'.join(irrelevant_type_list), case=False, na=False)]
    
    if file_type == 'excavator':
        # 对还未标记类型的条目进行标记，搜索 ‘crawler excavator’ 关键字
        df.loc[(df['type']=='UNKNOWN')&(df['description2'].str.contains('crawler excavator',case=False)),'type'] = 'EXCAVATOR'

        # 对还未标记类型的条目进行标记，搜索 ‘wheel excavator’ 关键字
        df.loc[(df['type']=='UNKNOWN')&(df['description2'].str.contains('wheel excavator',case=False))&(df['description2'].str.contains('wheel',case=False)),'type'] = 'WHEEL EXCAVATOR'

        # 对还未标记类型的条目进行标记，搜索 ‘AMPHIBIOUS EXCAVATOR’ 关键字
        df.loc[(df['type']=='UNKNOWN')&(df['description2'].str.contains('AMPHIBIOUS',case=False)),'type'] = 'AMPHIBIOUS EXCAVATOR'

        # 对存在‘tire’关键字的条目进行标记
        df.loc[(df['description2'].str.contains('tire',case=False)),'type']='WHEEL EXCAVATOR'
        
    elif file_type == 'crane':
        df.loc[(df['description2'].str.contains('Wheel|tire',case=False))&(df['type']=='UNKNOWN'),'type'] = 'WHEELED CRANE'
        df.loc[(df['description2'].str.contains('rough',case=False))&(df['type']=='UNKNOWN'),'type'] = 'ROUGH-TERRAIN CRANE'
        df.loc[(df['description2'].str.contains('crawler',case=False))&(df['type']=='UNKNOWN'),'type'] = 'CRAWLER CRANE'
        df.loc[(df['description2'].str.contains('crawler',case=False))&(df['description2'].str.contains('telescopic',case=False)),'type'] = 'CRAWLER CRANE (TELESCOPIC_BOOM)'

    # 标记新旧机
    PROD_YEAR = np.arange(1950,2019,1) # 1950-2019年之间的设备都算旧机
    condition_list = ['USED', 'SECOND HAND', 'SECONDHAND', 'OLD', '2ND HAND', 'REFURBISH'] # 新旧关键字列表
    condition_list += [str(year) for year in PROD_YEAR] # 合并两个列表

    for index, row in df.iterrows():
        description = row['description2']
        condition_temp_list = [1 if description.split(' ').count(condition_key.upper()) > 0 else 0 for condition_key in condition_list]
        condition = 'used' if sum(condition_temp_list) > 0 else 'new'
        df.at[index, 'new/used'] = condition

    # 因为年份可能会有误判，需要进行二次判断
    df.loc[(df['description2'].str.contains('NEW|UNUSED',case=False))&(df['new/used']=='used'),'new/used']='new'

    # 对CKD，SKD和partial的条目进行标记
    partial_list = ['CKD', 'SKD', 'partial']
    df.loc[df['description2'].str.contains('|'.join(partial_list), case=False, na=False),'remark']='Parts'
    
    return df, error_rows_index


# 通过regex对没有匹配的数据进行标记
def search_regex(df_original, condition, search_col_name, df_regex_original, filter_brand=True, mark_load=True):
    import pandas as pd
    import numpy as np
    import re
    df = df_original.copy() # 新建一个副本，不修改原始表格
    
    for index, row in df.loc[condition].iterrows():
        description = row[search_col_name]
        brand = row['brand']
        
        if filter_brand:
            if brand != 'UNKNOWN':  # 如果品牌已经匹配，但无型号
                df_regex = df_regex_original[df_regex_original['brand'] == brand]  # 对照表将只会在该品牌下循环
            else:
                df_regex = df_regex_original  # 在品牌为UNKNOWN的情况下，循环整个对照表
        else:
            df_regex = df_regex_original
            
        for model_regex in df_regex['model_regex']:
            model_match = re.findall(model_regex, description)
            if model_match:                   
                df.at[index, 'model'] = max(model_match, key=len) # 取最长型号
                df.at[index, 'brand'] = df_regex.loc[df_regex['model_regex']==model_regex,'brand'].values[0]
                df.at[index, 'type'] = df_regex.loc[df_regex['model_regex']==model_regex,'category'].values[0]

                if len(model_match)>1: # regex匹配结果大于1时
                    # 以下两行代码会保留多个匹配到的规律；但实际使用中发现底盘型号里的类似规律也会匹配到，此处选择只保留第一个，型号一般比底盘型号出现的靠前
                    # df.at[index, 'model'] = ', '.join(model_match)
                    # df.at[index, 'remark'] = '根据规律匹配，匹配的结果大于1个' # 这里可能会匹配到多个类似的规律，比如型号和底盘号近似的，需要单独检查，取长度最长的
                    if filter_brand:
                        df.at[index, 'remark'] = 'Keep the longest from the multiple matched' 
                    else:
                        df.at[index, 'remark'] = 'No brand in description, and keep the longest from the multiple matched' 
                else: # regex匹配结果唯一时
                    if filter_brand:
                        df.at[index, 'remark'] = 'Unique model match with regex' 
                    else:
                        df.at[index, 'remark'] = 'No brand in description, and unique model match with regex' 
                    
                if mark_load: # 在arguments里选择是否标记吨位
                    capacity_regex = df_regex.loc[df_regex['model_regex']==model_regex,'capacity_regex'].values[0]
                    numeric_part = re.search(capacity_regex, max(model_match, key=len))

                    if numeric_part: # 如果数字部分存在的话
                        starting_point = df_regex.loc[df_regex['model_regex']==model_regex,'starting_point'].values[0]
                        numeric_value = numeric_part.group(1)

                        if starting_point == 0: # 如果标记是0，取数字部分除10
                            capacity = float(numeric_value) / 10
                            df.at[index, 'capacity'] = capacity
                        elif starting_point == 1: # 如果标记是1，取从第二位开始的数字部分除10
                            numeric_value = numeric_part.group(1)[1:]
                            capacity = float(numeric_value) / 10
                        elif starting_point == 2: # 如果标记是2，说明吨位和型号数字没有关系，不做标记
                            capacity = 'TBD' # to be defined 需要单独查询
                        elif starting_point == 3: # 如果标记是3，直接标记数字部分
                            capacity = float(numeric_value) 
                        elif starting_point == -2: # 如果标记是-2，取第三位开始的数字部分
                            numeric_value = numeric_part.group(1)[2:]
                            capacity = float(numeric_value)
                            df.at[index, 'capacity'] = capcacity
                        else: # 如果标记是-1，取第二位开始的数字部分
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
    # 将未知型号，但有吊重、类型和品牌的条目，反向匹配数据中的对应型号
    for index,row in df.iterrows():
        brand = row['brand']
        model = row['model']
        capacity = row['capacity']
        model_type = row['type']
        model_info = df.loc[(df['brand']==brand)&(df['model']!='UNKNOWN'),['model','capacity','type']].drop_duplicates()
        # 吊装能力、品牌、类型已知，型号未知。从全数据表里反向匹配型号
        if capacity!='UNKNOWN' and brand!='UNKNOWN' and model=='UNKNOWN' and model_type!='UNKNOWN':
            if model_type in model_info['type'].unique(): # 如果类型存在于品牌中
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
    print(f"含有'partial'的条目数量：{(result['description2'].str.contains('partial', case=False) == True).sum()}")
    print(f"含有'party'的条目数量：{(result['description2'].str.contains('party', case=False) == True).sum()}")
    print(f"含有'part'的条目数量：{(result['description2'].str.contains('part', case=False) == True).sum()}")
    print(f"含有'assemble'的条目数量：{(result['description2'].str.contains('assemble', case=False) == True).sum()}")
    print(f"含有'SKD'的条目数量：{(result['description2'].str.contains('skd', case=False) == True).sum()}")
    

# 如果存在条款列，term参数需填写为True
def mark_outliers(df_original, term=True):
    df = df_original.copy()
    # 计算品牌型号单价的的中位数，如果存在贸易条款，要同一贸易条款下的进行判断
    # 型号是“UNKNOWN”的，异常值列标记“未知”；否则
    # 检查吨位是否在“参考单台重量” x ±20% 的范围内，如果不在范围内，异常值列标记“是”；否则
    # 检查价格是否符合同一型号的 中位数x ±20% 的范围内，如果不在范围内，异常值列标记“是”，否则标记“否”
    if term: # 包含贸易条款的情况下
        # 检查是否有必须存在的列名
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
    else: # 不包含贸易条款的情况下
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
    # 汇率需要单独填写在方程外面
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
    # 把吨位和重量转换成数字，便于计算，注意此处原本的string‘unknown’会被强行转化为空值    
    # 检查必须存在的列名
    required_columns = ['unit weight in ton']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"make sure the following column names are existed {required_columns}. Otherwise, please create the col with formular: weight in kg/qty/1000")
                
    df['capacity'] = pd.to_numeric(df['capacity'], errors='coerce')
    df['unit weight in ton'] = pd.to_numeric(df['unit weight in ton'], errors='coerce')
        
    # 取吨位和重量里最大的作为吨位上限
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



# 根据工况划分吨位区间 70吨以上的是矿挖
def define_excavator_load_type_interval(df):
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
    
    
    
# 更新regex对照表
def update_regex_df(df_regex, update_list):
    import pandas as pd
    # 更新表格格式如下
    # update_list = [{'brand': 'KOMATSU', 'model_regex': r'HB\d{3}-\d', 'capacity_regex':r'HB(\d+)', 'category':'EXCAVATOR', 'starting_point':0}]
    update_df = pd.DataFrame(update_list)
    df_regex = pd.concat([df_regex,update_df],axis=0,ignore_index=True)
    # 对照表去重
    df_regex = df_regex.drop_duplicates(subset=['brand','model_regex'])
    # 以model_regex的长度降序排列
    df_regex.sort_values(by='model_regex', key=lambda x: x.str.len(), ascending=False, inplace=True)
    df_regex = df_regex.reset_index(drop=True)
    return df_regex


# 检查列名是否一致
def check_col_names(df_col_names, ref_col_names):
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
#             df2.loc[index, ['model','capacity','type','remark']] = [zoomlion_model, load/1000, 'excavator'.upper(),'通过重量预估型号']
#         else:
#             pass


def matching_program_individual(df_original, condition, df_ref):
    df = df_original.copy()
    error_rows_index={} # 对程序运行过程中出现问题的条目进行单独标记，以便后期debug
    for index, row in df[condition].iterrows():
        brand = row['brand']
        product_description = row['description2']
        filtered_reference_df = df_ref[df_ref['brand']==brand]
        refrence_model_list = df_ref.loc[df_ref['brand']==brand,'model_ref'].values
        try:         
            # 该品牌存在型号的情况下
            if len(refrence_model_list) > 0:
                
                model_temp_list = []
                for model in refrence_model_list:
                    if model in product_description:
                        model_temp_list.append(model)
                        
                if len(model_temp_list) >= 1:
                    longest_model = max(model_temp_list, key=len)
                    df.at[index,'model']= longest_model #标记型号
                    df.loc[index,['capacity','hp','type']]= filtered_reference_df.loc[filtered_reference_df['model_ref']==longest_model, ['capacity','hp','type']].values[0] # 标记吨位和型号
                    df.at[index, 'remark'] = 'Fully match'

                else: # 如果没有匹配的型号
                    df.at[index,'model']= 'UNKNOWN' #标记未知
                    df.loc[index,['capacity','type']] = ['UNKNOWN','UNKNOWN'] # 标记吨位和型号未知
                    df.at[index, 'remark'] = 'Brands existed but without models'

            # 不存在品牌的情况下，直接标记，不反向搜索（通过型号判断品牌）
            else:
                df.at[index,'remark'] = 'Brands existed but without models'
        
        except Exception as e:
            error_rows_index[index] = str(e)
            df.at[index, 'remark'] = 'Error Occurred'
            
    return df, error_rows_index


def new_or_used(df,search_col):
    # 标记新旧机
    PROD_YEAR = np.arange(1950,2019,1) # 1950-2019年之间的设备都算旧机
    condition_list = ['USED', 'SECOND HAND', 'SECONDHAND', 'OLD', '2ND HAND', 'REFURBISH'] # 新旧关键字列表
    condition_list += [str(year) for year in PROD_YEAR] # 合并两个列表

    for index, row in df.iterrows():
        description = row[search_col]
        condition_temp_list = [1 if description.split(' ').count(condition_key.upper()) > 0 else 0 for condition_key in condition_list]
        condition = 'used' if sum(condition_temp_list) > 0 else 'new'
        df.at[index, 'new/used'] = condition

    # 因为年份可能会有误判，需要进行二次判断
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
    # 将数量和重量转换为数字格式便于计算
    df['qty'] = pd.to_numeric(df['qty'], errors='coerce')
    df['weight in kg'] = pd.to_numeric(df['weight in kg'], errors='coerce')

    # 计算单价和单位数量吨净重
    df['price in usd'] = df['amount in usd']/df['qty']
    df['unit weight in ton'] = df['weight in kg']/df['qty']/1000
    
    

def remove_price_outliers(df,amount):    
    # 先剔除总金额在10000美金以下的条目
    df = df[~(df['amount in usd']<amount)]

    # 删除单价在10000美金以下的条目
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