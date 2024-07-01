# earthmoving-mobile-crane-CRISP-DM-project
**Description of the project**: Customs data cleaning, mining and outlier definition. The aim is to automatically extract useful information from customs data for further analysis.

Customs data is a useful source for understanding the market of a specific region since it contains valuable information such as exporter, importer, business terms, product details, price, and et cetera, which provide aboundant insight to business. However, the data format differ from countries could be problematic, we can't simply extract the information based on specific rules. Therefore, I launched this project to automative my work.

My idea contains several sections:

1. **data inspection**: Unify col_names, units, currency, remove rows which contain irrelevant-keywords or abnormal prices;
2. **data pre-processing**: Upper case the data and remove special marks;
3. **matching the info from knowledge-base**: Matching the info in the raw data with the existed knowledge-base that scraped from the public-library;
4. **matching the info with regex knowledge-base**: Matching the rest of raw data with designed regex knowledge-base;
5. **backward tagging with exsisted info**: Infer labels based on other information in the data, as in the example that follows;

    <details>
    <summary>example</summary>
      
    | item | brand | model | capacity | type |
    |------|------|------|------|------|
    | A    | ✓    | ✓    | ✓    | ✓    |
    | B    | ✓    | ✘    | ✓    | ✓    |
    
    </details>

7. **mark the outliers**: Labelling out the rows with abnormal prices for further price analysis;
8. **categorize key indicators by interval**
9. **calculate the exchanging rate**
10. **update historical data**

Then, further analysis will ba made based on the cleaned data.

Here I attached my project notebook and the Py file for reference. The excel name in 824952 is the original dataframe. Enjoy yo!
