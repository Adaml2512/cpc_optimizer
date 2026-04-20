from typing import Dict, Any

import pandas as pd

def run(
    inputs: Dict[str, pd.DataFrame],
    parameters: Dict[str, Any],
    configs: Dict[str, Any]
) -> Dict[str, pd.DataFrame]:
    
    ##
    ## Add your logic here
    ## create and return outputs
    ##

    ## Get the inputs from the Python dictionaries
    my_input_table = inputs["my_input_table_name"]
    my_parameter = parameters["my_parameter_name"]

    ## Write some logic and return output tables

    return {
        "table1_display_name":table1_name_in_code,
        "table2_display_name":table2_name_in_code,
    }



if __name__ == "__main__":
    ## Create sample datasets in the data framework that your code uses (pandas, polars, spark, other).
    ## Or load in sample datasets

    ## Use this space to test your code locally on your computer.
    ## It isn't used in the green note.
    