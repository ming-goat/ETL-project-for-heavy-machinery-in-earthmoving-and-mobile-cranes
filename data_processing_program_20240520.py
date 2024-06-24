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


# 文字处理去掉特殊符号
def pre_processing(description):
    description = str(description)  # Convert to str if not already
    description = deal_special_brand(description)
    description = description.replace(',', ' ')  # 用空格替换逗号
    description = description.replace('(', ' ')  # 用空格替换左括号
    description = description.replace(')', ' ')  # 用空格替换右括号
    description = description.replace('[', ' ')  # 用空格替换左方括号
    description = description.replace(']', ' ')  # 用空格替换右方括号
    description = description.replace('/', '')  # 去掉斜杠号
    description = description.replace(':', ' ')  # 用空格替换冒号
    description = description.replace('*', '')  # 去掉星号
    description = description.replace(';', ' ')  # 用空格替换分号
    description = description.rstrip()  # 去除多余空格
    description = description.lstrip()  # 去除多余空格
    description = description.upper()  # Convert to uppercase
    return description


def data_preperation(df, df_ref):
    import re
    # 检查是否有必须存在的列名
    required_columns = ['供应商', '产品描述']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"错误提示: 请确保表格中包含如下列名 {required_columns}.")
        
    # 新建一个产品描述的副本以免影响原始数据
    df['货物描述'] = df['产品描述']
    
    # 统一供应商大写及去除特殊符号
    df['供应商'] = df.loc[:,'供应商'].apply(lambda x: pre_processing(x))
    # 去除多余的空格
    df['供应商'] = df['供应商'].apply(lambda x: re.sub(r'\s+', ' ', x))

    # 统一货物描述大写及去除特殊符号
    df['货物描述'] = df.loc[:,'货物描述'].apply(lambda x: pre_processing(x))
    # 去除多余的空格
    df['货物描述'] = df.loc[:,'货物描述'].apply(lambda x: re.sub(r'\s+', ' ', x))

    # 统一对照表大写及去除特殊符号
    df_ref['model_ref'] = df_ref.loc[:,'model'].apply(lambda x: pre_processing(x))
    # 去除多余的空格
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
    error_rows_index={} # 对程序运行过程中出现问题的条目进行单独标记，以便后期debug
    for index, row in df.iterrows():
        product_description = row['货物描述']
        supplier = row['供应商']
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
                df.at[index, '品牌'] = first_brand # 标记品牌
                filtered_reference_df = df_ref[df_ref['brand']==first_brand] # 根据匹配的型号筛选对照表，仅对筛选后的型号进行搜索，为型号匹配准备      
                unique_models = filtered_reference_df['model_ref'] # 筛选后品牌下的所有对照型号

                model_temp_list = []
                for model in unique_models:
                    if model in product_description:
                        model_temp_list.append(model)

                if len(model_temp_list) >= 1:
                    longest_model = max(model_temp_list, key=len)
                    df.at[index,'型号']= longest_model #标记型号
                    df.loc[index,['吨位','类型']]= filtered_reference_df.loc[filtered_reference_df['model_ref']==longest_model, ['capacity','type']].values[0] # 标记吨位和型号
                    df.at[index, '备注'] = '完全匹配'

                else: # 如果没有匹配的型号
                    df.at[index,'型号']= 'UNKNOWN' #标记未知
                    df.loc[index,['吨位','类型']] = ['UNKNOWN','UNKNOWN'] # 标记吨位和型号未知
                    df.at[index, '备注'] = '存在品牌，没有匹配型号'

            # 不存在品牌的情况下，直接标记，不反向搜索（通过型号判断品牌）
            else:
                df.loc[index,['品牌','型号','吨位','类型','备注']] = ['UNKNOWN','UNKNOWN','UNKNOWN','UNKNOWN','无品牌匹配']
        
        except Exception as e:
            error_rows_index[index] = str(e)
            df.at[index, '备注'] = 'Error Occurred'

    # BLOCK 2: 类型、新旧标记
    # 删除包含非相关关键字的条目
    irrelevant_type_list = ['CARRIER', 'TELESCOPLADER', 'HARBOUR', 'OPEN SHEET', 'STACK', 
                            'BOAT', 'BACKHOE', 'SKID', 'ROLLER', 'BENZ', 'TELEHANDLER', 
                            'LOADER', 'FORK', 'PAVER', 'STACKER', 'MATERIAL HANDLER', 
                            'BRIDGE', 'REACH', 'HANDER', 'GRABBER', 'GANTRY', 'BACK HOE', 
                            'PORT', 'MERCEDES', 'VİNCE', 'SPIDER', 'PIPE', 'HANDLING', 
                            'GLASS CRANE', 'LOAD', 'GRADER', 'GLASS CRANE', 'SPIDER']
    df = df.loc[~df['货物描述'].str.contains('|'.join(irrelevant_type_list), case=False, na=False)]
    
    if file_type == 'excavator':
        # 对还未标记类型的条目进行标记，搜索 ‘crawler excavator’ 关键字
        df.loc[(df['类型']=='UNKNOWN')&(df['货物描述'].str.contains('crawler excavator',case=False)),'类型'] = 'EXCAVATOR'

        # 对还未标记类型的条目进行标记，搜索 ‘wheel excavator’ 关键字
        df.loc[(df['类型']=='UNKNOWN')&(df['货物描述'].str.contains('wheel excavator',case=False))&(df['货物描述'].str.contains('wheel',case=False)),'类型'] = 'WHEEL EXCAVATOR'

        # 对还未标记类型的条目进行标记，搜索 ‘AMPHIBIOUS EXCAVATOR’ 关键字
        df.loc[(df['类型']=='UNKNOWN')&(df['货物描述'].str.contains('AMPHIBIOUS',case=False)),'类型'] = 'AMPHIBIOUS EXCAVATOR'

        # 对存在‘tire’关键字的条目进行标记
        df.loc[(df['货物描述'].str.contains('tire',case=False)),'类型']='WHEEL EXCAVATOR'
        
    elif file_type == 'crane':
        df.loc[(df['货物描述'].str.contains('Wheel|tire',case=False))&(df['类型']=='UNKNOWN'),'类型'] = 'WHEELED CRANE'
        df.loc[(df['货物描述'].str.contains('rough',case=False))&(df['类型']=='UNKNOWN'),'类型'] = 'ROUGH-TERRAIN CRANE'
        df.loc[(df['货物描述'].str.contains('crawler',case=False))&(df['类型']=='UNKNOWN'),'类型'] = 'CRAWLER CRANE'
        df.loc[(df['货物描述'].str.contains('crawler',case=False))&(df['货物描述'].str.contains('telescopic',case=False)),'类型'] = 'CRAWLER CRANE (TELESCOPIC_BOOM)'

    # 标记新旧机
    PROD_YEAR = np.arange(1950,2019,1) # 1950-2019年之间的设备都算旧机
    condition_list = ['USED', 'SECOND HAND', 'SECONDHAND', 'OLD', '2ND HAND', 'REFURBISH'] # 新旧关键字列表
    condition_list += [str(year) for year in PROD_YEAR] # 合并两个列表

    for index, row in df.iterrows():
        description = row['货物描述']
        condition_temp_list = [1 if description.split(' ').count(condition_key.upper()) > 0 else 0 for condition_key in condition_list]
        condition = '旧' if sum(condition_temp_list) > 0 else '新'
        df.at[index, '新旧'] = condition

    # 因为年份可能会有误判，需要进行二次判断
    df.loc[(df['货物描述'].str.contains('NEW|UNUSED',case=False))&(df['新旧']=='旧'),'新旧']='新'

    # 对CKD，SKD和partial的条目进行标记
    partial_list = ['CKD', 'SKD', 'partial']
    df.loc[df['货物描述'].str.contains('|'.join(partial_list), case=False, na=False),'备注']='散装件'
    
    return df, error_rows_index


# 通过regex对没有匹配的数据进行标记
def search_regex(df_original, condition, search_col_name, df_regex_original, filter_brand=True, mark_load=True):
    import pandas as pd
    import numpy as np
    import re
    df = df_original.copy() # 新建一个副本，不修改原始表格
    
    for index, row in df.loc[condition].iterrows():
        description = row[search_col_name]
        brand = row['品牌']
        
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
                df.at[index, '型号'] = max(model_match, key=len) # 取最长型号
                df.at[index, '品牌'] = df_regex.loc[df_regex['model_regex']==model_regex,'brand'].values[0]
                df.at[index, '类型'] = df_regex.loc[df_regex['model_regex']==model_regex,'category'].values[0]

                if len(model_match)>1: # regex匹配结果大于1时
                    # 以下两行代码会保留多个匹配到的规律；但实际使用中发现底盘型号里的类似规律也会匹配到，此处选择只保留第一个，型号一般比底盘型号出现的靠前
                    # df.at[index, '型号'] = ', '.join(model_match)
                    # df.at[index, '备注'] = '根据规律匹配，匹配的结果大于1个' # 这里可能会匹配到多个类似的规律，比如型号和底盘号近似的，需要单独检查，取长度最长的
                    if filter_brand:
                        df.at[index, '备注'] = '根据规律，有多个匹配结果，但只保留最长型号' 
                    else:
                        df.at[index, '备注'] = '描述中未找到对应品牌，根据规律匹配型号，有多个匹配结果，但只保留最长型号' 
                else: # regex匹配结果唯一时
                    if filter_brand:
                        df.at[index, '备注'] = '根据规律匹配，型号唯一' 
                    else:
                        df.at[index, '备注'] = '描述中未找到对应品牌，根据规律匹配型号，型号唯一' 
                    
                if mark_load: # 在arguments里选择是否标记吨位
                    capacity_regex = df_regex.loc[df_regex['model_regex']==model_regex,'capacity_regex'].values[0]
                    numeric_part = re.search(capacity_regex, max(model_match, key=len))

                    if numeric_part: # 如果数字部分存在的话
                        starting_point = df_regex.loc[df_regex['model_regex']==model_regex,'starting_point'].values[0]
                        numeric_value = numeric_part.group(1)

                        if starting_point == 0: # 如果标记是0，取数字部分除10
                            capacity = float(numeric_value) / 10
                            df.at[index, '吨位'] = capacity
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
                            df.at[index, '吨位'] = capcacity
                        else: # 如果标记是-1，取第二位开始的数字部分
                            numeric_value = numeric_part.group(1)[1:]
                            capacity = float(numeric_value)
                            df.at[index, '吨位'] = capacity
                    else:
                        df.at[index, '吨位'] = 'UNKNOWN'                
                        
    return df


# def search_capacity(df_original, condition, search_col, capacity_regex):
#     import re
#     # capacity_regex = r'\b(\d+)\s*(?:METRIC\s*)?TONS?\b'
#     # (?:METRIC\s*)?: Matches an optional 'METRIC' followed by optional whitespace characters. 
#     # The (?: ... ) is a non-capturing group.
#     df = df_original.copy()
#     for index, row in df.loc[condition].iterrows():
#         description = row[search_col]
#         model_match = re.findall(capacity_regex, description)
#         if model_match:
#             numeric_part = re.search(r'(\d+)', model_match[0])
#             if numeric_part:
#                 df.at[index, '吨位'] = numeric_part.group(1)
#                 df.at[index, '备注'] = '描述中包含吊装载重能力描述'
#     return df


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
            df.at[index, '吨位'] = capacity
            df.at[index, '备注'] = '描述中包含吊装载重能力描述'
    return df


def mark_unknown_model_with_exsisted_lifting_capacity(df_original):
    df = df_original.copy()
    # 将未知型号，但有吊重、类型和品牌的条目，反向匹配数据中的对应型号
    for index,row in df.iterrows():
        brand = row['品牌']
        model = row['型号']
        capacity = row['吨位']
        model_type = row['类型']
        model_info = df.loc[(df['品牌']==brand)&(df['型号']!='UNKNOWN'),['型号','吨位','类型']].drop_duplicates()
        # 吊装能力、品牌、类型已知，型号未知。从全数据表里反向匹配型号
        if capacity!='UNKNOWN' and brand!='UNKNOWN' and model=='UNKNOWN' and model_type!='UNKNOWN':
            if model_type in model_info['类型'].unique(): # 如果类型存在于品牌中
                model_info = model_info[model_info['类型']==model_type]
                # threshhold within +-5%
                for exsisted_capacity in model_info['吨位']:
                    exsisted_capacity = float(exsisted_capacity)
                    capacity = float(capacity)
                    if exsisted_capacity > capacity*0.95 and exsisted_capacity < capacity*1.05:
                        df.at[index,'型号'] = model_info.loc[model_info['吨位']==exsisted_capacity,'型号'].iloc[0]
                        df.at[index,'备注'] = '描述中包含吊装载重能力描述，型号根据已知型号反向匹配'
    return df


def check_parts(result):
    print(f"含有'partial'的条目数量：{(result['货物描述'].str.contains('partial', case=False) == True).sum()}")
    print(f"含有'party'的条目数量：{(result['货物描述'].str.contains('party', case=False) == True).sum()}")
    print(f"含有'part'的条目数量：{(result['货物描述'].str.contains('part', case=False) == True).sum()}")
    print(f"含有'assemble'的条目数量：{(result['货物描述'].str.contains('assemble', case=False) == True).sum()}")
    print(f"含有'SKD'的条目数量：{(result['货物描述'].str.contains('skd', case=False) == True).sum()}")
    

# 如果存在条款列，term参数需填写为True
def mark_outliers(df_original, term=True):
    df = df_original.copy()
    # 计算品牌型号单价的的中位数，如果存在贸易条款，要同一贸易条款下的进行判断
    # 型号是“UNKNOWN”的，异常值列标记“未知”；否则
    # 检查吨位是否在“参考单台重量” x ±20% 的范围内，如果不在范围内，异常值列标记“是”；否则
    # 检查价格是否符合同一型号的 中位数x ±20% 的范围内，如果不在范围内，异常值列标记“是”，否则标记“否”
    if term: # 包含贸易条款的情况下
        # 检查是否有必须存在的列名
        required_columns = ['交易条款']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"错误提示: 请确保表格中包含如下列名 {required_columns}.")
                
        midian_price = df.groupby(['品牌','型号','交易条款'])['美元单价'].agg('median').reset_index()
        df['异常值列标记'] = ["未知" 
         if model_label == 'UNKNOWN' or load == 'UNKNOWN' or used_new == '旧'
         else "是" if unit_price >= 1.2*(midian_price[(midian_price['品牌']==brand) & (midian_price['型号']==model_label) & (midian_price['交易条款']==term)]['美元单价'].values[0])
         else "是" if unit_price <= 0.8*(midian_price[(midian_price['品牌']==brand) & (midian_price['型号']==model_label)]['美元单价'].values[0])
         else "否" 
         for brand, model_label, load, unit_price, used_new, term in zip(df['品牌'], df['型号'], df['吨位'], df['美元单价'], df['新旧'], df['交易条款'])]
    else: # 不包含贸易条款的情况下
        midian_price = df.groupby(['品牌', '型号'])['美元单价'].agg('median').reset_index()
        df['异常值列标记'] = ["未知" 
         if model_label == 'UNKNOWN' or load == 'UNKNOWN' or used_new == '旧'
         else "是" if unit_price >= 1.2*(midian_price[(midian_price['品牌']==brand) & (midian_price['型号']==model_label)]['美元单价'].values[0])
         else "是" if unit_price <= 0.8*(midian_price[(midian_price['品牌']==brand) & (midian_price['型号']==model_label)]['美元单价'].values[0])
         else "否" 
         for brand, model_label, load, unit_price, used_new in zip(df['品牌'], df['型号'], df['吨位'], df['美元单价'], df['新旧'])]
    
    return df


# 计算人民币汇率单价
def convert_usd_to_cny(df_original, rate_dict): 
    # 汇率需要单独填写在方程外面
    # rate_dict = {2023: {1: 6.7604, 2: 6.9519, 3: 6.8717, 4: 6.924, 5: 7.0821, 6: 7.2258, 7: 7.1305, 8: 7.1811, 9: 7.1798, 10: 7.1779, 11: 7.1018, 12: 7.0827},
             # 2024: {1: 7.1039, 2: 7.1036, 3: 7.0950, 4:7.1063, 5:7.1088}}
    import pandas as pd    
    df = df_original.copy()
    df['日期'] = pd.to_datetime(df['日期'])
    for year, month_dict in rate_dict.items():
        for month, rate in month_dict.items():
            # Filter DataFrame for the current year and month
            filtered_df = df[(df['日期'].dt.year == year) & (df['日期'].dt.month == month)]

            # Check if the necessary columns exist
            required_columns = ['日期', '美元单价', '美元金额']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"The DataFrame is missing one or more required columns from {required_columns}.")

            # Convert USD price and amount to RMB
            df.loc[(df['日期'].dt.year == year) & (df['日期'].dt.month == month), '人民币单价'] = filtered_df['美元单价'] * rate
            df.loc[(df['日期'].dt.year == year) & (df['日期'].dt.month == month), '人民币金额'] = filtered_df['美元金额'] * rate
    return df


def define_load_interval(df, file_type = 'excavator', load_interval = 50):
    import pandas as pd
    # 把吨位和重量转换成数字，便于计算，注意此处原本的string‘unknown’会被强行转化为空值    
    # 检查必须存在的列名
    required_columns = ['单位数量吨净重']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"请确保数据表里存在 {required_columns} 列。若没有，请新建列并按照公式：重量（kg）/数量/1000 进行计算。")
                
    df['吨位'] = pd.to_numeric(df['吨位'], errors='coerce')
    df['单位数量吨净重'] = pd.to_numeric(df['单位数量吨净重'], errors='coerce')
        
    # 取吨位和重量里最大的作为吨位上限
    if file_type == 'excavator':
        max_load = max(df.loc[pd.notna(df['吨位']), '吨位'].max(),df.loc[df['单位数量吨净重']!=0, '单位数量吨净重'].max())
    if file_type == 'crane':
        max_load = df.loc[pd.notna(df['吨位']), '吨位'].max()
        
    upper_limit_index = int(round(max_load/load_interval, 0) + 1)
    
    if file_type == 'excavator':
        for index, row in df.iterrows():
            load = row['吨位']
            weight = row['单位数量吨净重']

            if pd.notna(load) and load!=0:
                evaluate_value = load
            else:
                evaluate_value = weight
                
            if evaluate_value == 0:
                df.at[index, '吨位范围'] = 'UNKNOWN'
            elif evaluate_value < 5:
                df.at[index, '吨位范围'] = '<5T'
            elif evaluate_value>=5 and evaluate_value<10:
                df.at[index, '吨位范围'] = '5-10T'
            else:
                for i in range(1, upper_limit_index):
                    lower_bound = i * load_interval
                    upper_bound = (i + 1) * load_interval
                    if evaluate_value >= lower_bound and evaluate_value < upper_bound:
                        df.at[index, '吨位范围'] = f'{lower_bound}-{upper_bound}T'
                    i = i+1
            
    if file_type == 'crane':
        for index, row in df.loc[pd.notna(df['吨位'])].iterrows():
            load = row['吨位']

            if load == 0:
                df.at[index, '吨位范围'] = 'UNKNOWN'
            elif load < load_interval:
                df.at[index, '吨位范围'] = f'<{load_interval}T'
            else:
                for i in range(1, upper_limit_index + 1):
                    lower_bound = i * load_interval
                    upper_bound = (i + 1) * load_interval
                    if load >= lower_bound and load < upper_bound:
                        df.at[index, '吨位范围'] = f'{lower_bound}-{upper_bound}T'
                    i = i + 1

    df['吨位'] = df['吨位'].fillna('UNKNOWN')



# 根据工况划分吨位区间 70吨以上的是矿挖
def define_excavator_load_type_interval(df):
    import pandas as pd
    for index, row in df.iterrows():
        load = row['吨位']
        weight = row['单位数量吨净重']
        
        if load != 'UNKNOWN' and load!=0:
            evaluate_value = load
        else:
            evaluate_value = weight
            
        if evaluate_value == 0 or pd.isna(evaluate_value):
            df.loc[index, '吨位类型划分'] = 'UNKNOWN'
        elif evaluate_value < 5:
            df.loc[index, '吨位类型划分'] = '<5T'
        elif evaluate_value>=5 and evaluate_value<10:
            df.loc[index, '吨位类型划分'] = '5-10T'
        elif evaluate_value>=10 and evaluate_value<30:
            df.loc[index, '吨位类型划分'] = '10-30T'
        elif evaluate_value>=30 and evaluate_value<70:
            df.loc[index, '吨位类型划分'] = '30-70T'
        elif evaluate_value>=70 and evaluate_value<90:
            df.loc[index, '吨位类型划分'] = '70-90T'
        else:
            df.loc[index, '吨位类型划分'] = '≥90T'
    
    
    
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
        print('列名一致')
        
        
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

# for index, row in df2.loc[df2['备注'].isin(['无品牌匹配','存在品牌，没有匹配型号'])&(df2['品牌']=='zoomlion'.upper())].iterrows():
#     weight = row['重量']
#     for load in zoomlion_dict.values():
#         if weight >= load*0.95 and weight < load*1.05:
#             zoomlion_model = get_key_from_value(zoomlion_dict, load)
#             df2.loc[index, ['型号','吨位','类型','备注']] = [zoomlion_model, load/1000, 'excavator'.upper(),'通过重量预估型号']
#         else:
#             pass


def matching_program_individual(df_original, condition, df_ref):
    df = df_original.copy()
    error_rows_index={} # 对程序运行过程中出现问题的条目进行单独标记，以便后期debug
    for index, row in df[condition].iterrows():
        brand = row['品牌']
        product_description = row['货物描述']
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
                    df.at[index,'型号']= longest_model #标记型号
                    df.loc[index,['吨位','马力','类型']]= filtered_reference_df.loc[filtered_reference_df['model_ref']==longest_model, ['capacity','hp','type']].values[0] # 标记吨位和型号
                    df.at[index, '备注'] = '完全匹配'

                else: # 如果没有匹配的型号
                    df.at[index,'型号']= 'UNKNOWN' #标记未知
                    df.loc[index,['吨位','类型']] = ['UNKNOWN','UNKNOWN'] # 标记吨位和型号未知
                    df.at[index, '备注'] = '存在品牌，没有匹配型号'

            # 不存在品牌的情况下，直接标记，不反向搜索（通过型号判断品牌）
            else:
                df.at[index,'备注'] = '存在品牌，没有匹配型号'
        
        except Exception as e:
            error_rows_index[index] = str(e)
            df.at[index, '备注'] = 'Error Occurred'
            
    return df, error_rows_index


def new_or_used(df,search_col):
    # 标记新旧机
    PROD_YEAR = np.arange(1950,2019,1) # 1950-2019年之间的设备都算旧机
    condition_list = ['USED', 'SECOND HAND', 'SECONDHAND', 'OLD', '2ND HAND', 'REFURBISH'] # 新旧关键字列表
    condition_list += [str(year) for year in PROD_YEAR] # 合并两个列表

    for index, row in df.iterrows():
        description = row[search_col]
        condition_temp_list = [1 if description.split(' ').count(condition_key.upper()) > 0 else 0 for condition_key in condition_list]
        condition = '旧' if sum(condition_temp_list) > 0 else '新'
        df.at[index, '新旧'] = condition

    # 因为年份可能会有误判，需要进行二次判断
    df.loc[(df[search_col].str.contains('NEW|UNUSED',case=False))&(df['新旧']=='旧'),'新旧']='新'
    
    
    
unit_regex_eng = r'(\d+)\s*UNITS'

def extract_units(df, search_col, keywords, regex):
    for index,row in df[df[search_col].str.contains(keywords,case=False)].iterrows():
        description = row[search_col]
        match = re.search(regex, description)
        if match:
            qty = int(match.group(1))
            df.loc[index,['数量','备注']]= [qty,'描述含数量关键词']
                
                

def unify_qty_weight(df):
    # 将数量和重量转换为数字格式便于计算
    df['数量'] = pd.to_numeric(df['数量'], errors='coerce')
    df['重量'] = pd.to_numeric(df['重量'], errors='coerce')

    # 计算单价和单位数量吨净重
    df['美元单价'] = df['美元金额']/df['数量']
    df['单位数量吨净重'] = df['重量']/df['数量']/1000
    
    

def remove_price_outliers(df,amount):    
    # 先剔除总金额在10000美金以下的条目
    df = df[~(df['美元金额']<amount)]

    # 删除单价在10000美金以下的条目
    df = df[df['美元单价']>=amount]
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
        description = row['产品描述']
    
        match = re.search(r'\b(ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN)\b', description, re.IGNORECASE)
        if match:
            number_word = match.group(1).upper()
            qty = number_words_to_digits[number_word]
            df.loc[index, ['数量','备注']] = [qty, '描述中存在数量']
            
    return df