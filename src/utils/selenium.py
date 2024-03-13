import os
import pandas as pd
from pathlib import Path
import time

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

import utils.consts as consts


def setup_driver():

    options = webdriver.ChromeOptions()

    options.add_experimental_option("prefs", {
        "download.default_directory": str(consts.PATH_PROJECT_OVERSAMPLING),
        # "download.directory_upgrade": True,
        # "download.prompt_for_download": False,
    })
    options.add_argument("--headless=new")
    webdriver_service = Service('/var/tmp/chromedriver/chromedriver')
    driver = webdriver.Chrome(options=options, service=webdriver_service)

    print(driver)

    return driver


def download_steno_cvd_results(df_patient_data, path_df_patient_data, type_over, id_label, seed):

    driver = setup_driver()
    driver.get(consts.URL_STENO_CALCULATOR)
    driver.find_element(By.XPATH, '/html/body/nav/div/ul/li[2]/a').click()

    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.XPATH, '//*[@id="file"]'))
    ).send_keys(path_df_patient_data)

    time.sleep(10)
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, '//*[@id="downloadData"]'))
    ).click()
    time.sleep(10)
    driver.close()

    os.rename(os.path.join(consts.PATH_DOWNLOAD_DIR, 'stenoRiskReport.csv'),
              os.path.join(consts.PATH_DOWNLOAD_DIR, 'steno_risk_report_{}_{}_seed_{}.csv'.format(type_over, id_label, seed)))

    path_file_steno_results = Path.joinpath(consts.PATH_PROJECT_OVERSAMPLING,
                                            'steno_risk_report_{}_{}_seed_{}.csv'.format(type_over, id_label, seed))

    return pd.read_csv(str(path_file_steno_results))

