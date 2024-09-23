import pandas as pd
import openai
import json
from pathlib import Path

def load_patient_data_xmap02(patient_data_xls: str):
    patient_data = pd.read_excel(patient_data_xls)
    columns_dict = {
        'Patient ID': int,
        'Age': int,
        'Sex': int,
        'Diabetes' : int,
        'Hereditary' : str,
        'Hypertension' : int,
        'Migraine' : int,
        'Smoking' : int,
        'VA' : float,
        'Unilateral' : int,
        'SE' : float,
        "Gonio_Shaffer" : int,
        "Gonio_Pigment" : int,
        "CD" : float,
        "CCT" : float,
        "OCT" : float,
        "Eye" : str,
        "IOP_Diagnosis" : float,
        "IOP_3" : float,
        "Catop_0" : int,
        "MD_Diagnosis" : float,
        "VFI_Diagnosis" : float,
        "MD_3" : float,
        "VFI_3" : float,
        "ROP_Percent" : float,
        "Prog" : int,
        "GPA" : str,
        "Null" : int,
        "Medicine_Count" : int,
        "SLT" : int,
        "Catop_3" : int,
    }
    #remove the last 6 columns
    clean_patient_data = patient_data.iloc[:, :-6]
    clean_patient_data.columns = columns_dict.keys()
    #drop the null column
    #convert the columns to the correct data types, with any nan values converted to 0 if the column is an integer or float and none if the column is string
    for column, dtype in columns_dict.items():
        if dtype == int:
            clean_patient_data[column] = clean_patient_data[column].fillna(0).astype(int)
        elif dtype == float:
            clean_patient_data[column] = clean_patient_data[column].fillna(0).astype(float)
        elif dtype == str:
            clean_patient_data[column] = clean_patient_data[column].fillna('None').astype(str)
    clean_patient_data = clean_patient_data.drop('Null', axis=1)

    def column_specific_processing(data:pd.DataFrame):
        
        return data
    
    clean_patient_data = column_specific_processing(clean_patient_data)
    #get the medicine list column from patient_data
    medicine_list = patient_data['Beh']
    #remove spaces and split the items based on "+"
    medicine_list = medicine_list.str.replace(' ', '').str.split('+')
    def condense_medicine_list(medicine_list:pd.DataFrame):
        #extract the medicine lists from each row and join into a single list
        flattened_list = [item.lower() for sublist in medicine_list.dropna() for item in sublist]
        unique_medicines = list(set(flattened_list))
        #sort the list alphabetically
        unique_medicines.sort()
        #remove any items called "slt" followed by any number of numbers, do not use re
        unique_medicines = [item for item in unique_medicines if not item.startswith('slt')]
        return unique_medicines
    def get_full_medicine_name(medicine_name:str):
        import openai
        import json
        #import keychain from keys
        keychain = Path("keys", ".keychain")
        keys = json.load(open(keychain))
        #openai.api_key = "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"
        openai.api_key = keys['openai']['xmap_2']
        prompt = f"The shortened name '{medicine_name}' refers to a medicine commonly prescribed for treating glaucoma in Sweden. Your response should be only the name of the medicine, as found on FASS.se, followed by a colon and then a python-format list of the active substances."
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_tokens=50
        )
        full_name = response.choices[0].message.content.strip()
        #remove any punctuation at the end of the response
        full_name = full_name.rstrip('.,')
        return full_name
    unique_medicines = condense_medicine_list(medicine_list)
    full_name_dict = {}
    for medicine in unique_medicines:
        full_name_entry = get_full_medicine_name(medicine)
        entry = {full_name_entry.split(':')[0].strip() : full_name_entry.split(':')[1]}
        full_name_dict[medicine] = entry
    active_ingredients_dict = {}
    for medicine in full_name_dict:
       name = next(iter(full_name_dict[medicine].keys()))
       active_ingredients = next(iter(full_name_dict[medicine].values()))
       active_ingredients = active_ingredients.lower().replace('[', '').replace(']', '').replace(' ', '').replace("'", '').replace('"', '').split(',')
       active_ingredients_dict[medicine] = active_ingredients

    #for each row, add a list of active ingredients based on the medicines in the medicine list series object
    ingredient_list = pd.Series()
    for row in medicine_list:
        #if row is nan, ingredents is "None"
        if not isinstance(row, list):
            ingredients = ['None']
            ingredient_list = pd.concat([ingredient_list, pd.Series([ingredients])])
            continue
        else:
            print(row)
            #ignore any starting with slt
            row = [item for item in row if not item.lower().startswith('slt')]
            ingredients = []
            for medicine in row:
                ingredients.extend(active_ingredients_dict[medicine.lower()])
            ingredient_list = pd.concat([ingredient_list, pd.Series([ingredients])])
    ingredient_list = ingredient_list.reset_index(drop=True)
        
    full_patient_data = clean_patient_data.copy()
    full_patient_data['Medicine_List'] = medicine_list
    full_patient_data['Active_Ingredients'] = ingredient_list
    full_patient_data.to_excel('data/02_patient_data_cleaned.xlsx', index=False)

    return full_patient_data

patient_data = pd.read_excel('data/02_patient_data.xlsx')
clean_patient_data = load_patient_data_xmap02('data/02_patient_data.xlsx')
