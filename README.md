# Pump-it-Up
Data science challenge hosted by drivendata

Using data from Taarifa and the Tanzanian Ministry of Water, can you predict which pumps are functional, which need some repairs, and which don't work at all? This is an intermediate-level practice competition. Predict one of these three classes based on a number of variables about what kind of pump is operating, when it was installed, and how it is managed. A smart understanding of which waterpoints will fail can improve maintenance operations and ensure that clean, potable water is available to communities across Tanzania.

## Features of the dataset
The goal is to predict the operating condition of a waterpoint for each record in the dataset. We are provided the following set of information about the waterpoints:  
    - **amount_tsh** - Total static head (amount water available to waterpoint)  
    - **date_recorded** - The date the row was entered  
    - **funder** - Who funded the well  
    - **gps_height** - Altitude of the well  
    - **installer** - Organization that installed the well  
    - **longitude** - GPS coordinate  
    - **latitude** - GPS coordinate  
    - **wpt_name** - Name of the waterpoint if there is one  
    - **num_private** -  
    - **basin** - Geographic water basin  
    - **subvillage** - Geographic location  
    - **region** - Geographic location  
    - **region_code** - Geographic location (coded)  
    - **district_code** - Geographic location (coded)  
    - **lga** - Geographic location  
    - **ward** - Geographic location  
    - **population** - Population around the well  
    - **public_meeting** - True/False  
    - **recorded_by** - Group entering this row of data  
    - **scheme_management** - Who operates the waterpoint  
    - **scheme_name** - Who operates the waterpoint  
    - **permit** - If the waterpoint is permitted  
    - **construction_year** - Year the waterpoint was constructed  
    - **extraction_type** - The kind of extraction the waterpoint uses  
    - **extraction_type_group** - The kind of extraction the waterpoint uses  
    - **extraction_type_class** - The kind of extraction the waterpoint uses  
    - **management** - How the waterpoint is managed  
    - **management_group** - How the waterpoint is managed  
    - **payment** - What the water costs  
    - **payment_type** - What the water costs  
    - **ater_quality** - The quality of the water  
    - **quality_group** - The quality of the water  
    - **quantity** - The quantity of water  
    - **quantity_group** - The quantity of water  
    - **source** - The source of the water  
    - **source_type** - The source of the water  
    - **source_class** - The source of the water  
    - **waterpoint_type** - The kind of waterpoint  
    - **waterpoint_type_group** - The kind of waterpoint  

## Distribution of Labels
The labels in this dataset are simple. There are three possible values:  
    - **functional** - the waterpoint is operational and there are no repairs needed  
    - **functional needs repair** - the waterpoint is operational, but needs repairs  
    - **non functional** - the waterpoint is not operational  

## Preprocessing of the data
The preprocessing is done using the function preproecessing_data in the utils file.  
Many features are highly correlated with others (like quality and quality_group). To avoid leakage we remove the following columns: "recorded_by", "lga", "num_private", "region_code", "district_code", "id", "ward", "scheme_name", "wpt_name", "extraction_type_class", 
               "extraction_type_group", "payment_type", "management", "water_quality", "quantity_group", "source", "source_class",
               "waterpoint_type_group", "management_group", "population", "subvillage", "key", "region", "date_recorded".  
NA values are replaced by default values and we try to fill missing GPS locations by looking at wells in the same village.  
Categorical features are one-hot encoded.  
Two features are added: month_recorded and days_since_recorded (from Jan 1st 2014)
Features gpds_height, longitude, latitude and construction_year are rescaled to [0,1]

## Classification
Decision trees from the library lightgbm are used. They are fast to learn and give better accuracy than xgboost trees.
8 trees, trained on different slices of the dataset are used to predict the final state of each pump. Parameters of these trees were obtained through a grid search maximazing the accuracy. In this problem, the score is the number of correct predictions over the total number of predictions.   
Once trained, we get a score on the hidden dataset of **0.8181**, which ranks us 914/10179 **(top 9%)** in October 2020.

## Architecture of the project
This projects is structured like this:
    - data/: contains the training set, testing set, an example of submission format, the submission file and the folder preprocessing
    - lib/:
        utils.py: contains the preprocessing of the data
        multi_trees.py: implements the training of trees and creates the submission file
    - PumpItUp-EDA.html: report generated by pandas-profiling to quickly explore the dataset
