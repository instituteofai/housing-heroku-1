from pipeline import get_pipeline

def get_prepared_data(payload):
    # num_attribs = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
    #     'total_bedrooms', 'population', 'households', 'median_income']

    full_pipeline = get_pipeline()

    prepared_data = full_pipeline.transform(payload)
    return prepared_data
