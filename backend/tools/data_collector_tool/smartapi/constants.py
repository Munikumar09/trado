from app import ROOT_DIR

NIFTY_500_STOCK_LIST_PATH = f"{ROOT_DIR}/data/nse/ind_nifty500list.csv"
DATA_DOWNLOAD_PATH = f"{ROOT_DIR}/data/historical_data/stocks"
DATA_STARTING_DATES_PATH = f"{ROOT_DIR}/data/smart_api/data_starting_dates.json"
HISTORICAL_STOCK_DATA_URL = "http://127.0.0.1:8000/smart-api/equity/history/"
