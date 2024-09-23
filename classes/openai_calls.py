import openai
import pandas as pd
class xmap_openai():
    """
    A class to interact with the OpenAI API for the xMAP project.

    Attributes:
        api_key (str): The OpenAI API key.
    """

    def __init__(self, api_key: str):
        """
        Initializes the xmap_openai class with the provided API key.

        Args:
            api_key (str): The OpenAI API key.
        """
        self.api_key = api_key
        openai.api_key = api_key

    def get_full_medicine_name(self, medicine_name:str):
        import openai
        import json
        #import keychain from keys
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
    
    def column_classification(self, column:pd.Series):
        """
        Classifies the type of data in a given column, ordinal, continuous, or categorical.
        """
        unique_values = column.unique()
        num_unique = len(unique_values)
        if num_unique == 1:
            return 'constant'
        elif num_unique == 2:
            return 'binary'
        elif num_unique == len(column):
            return 'identifier'
        else:
            return 'undetermined'
        
    def evaluate_column(self, column:pd.Series):
        unique_values = column.unique()
        colname = column.name
        prompt = f"Given the following values from the column '{colname}', classify the type of data: {unique_values}. Your response should be a single word, one of the following: 'categorical', 'ordinal', 'continuous'."
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_tokens=50
        )
        return response

    def order_columns(self, data:pd.DataFrame):
        column_class = {}
        for column in data: 
            classification = self.column_classification(data[column])
            if classification != 'undetermined':
                column_class[column] = classification
            else:
                response = self.evaluate_column(data[column])
                column_class[column] = response.choices[0].message.content.strip().lower()

        #sort the columns of the original data to group based on the column classification
        sorted_columns = sorted(column_class, key=lambda x: column_class[x])
            # Reorder the DataFrame columns based on the sorted classification
        structured_data = data[sorted_columns]
        
        # Sort the column_class dictionary by its values
        sorted_column_class = dict(sorted(column_class.items(), key=lambda item: item[1]))

        return structured_data, sorted_column_class

        
