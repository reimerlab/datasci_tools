
web_autoation_tools
#########
tools for helping with web scrapping and web automation using tools like BeautifulSoup and Selenium


Downloading the Chrome Driver For Web Automation
#################
In order for the Selenium package to work with Chrome to enable web interaction the 
chrome driver needs to be downloaded and extracted and placed in a known location for the 
script to use (placing in Downloads folder is assumed otherwise the path needs to be specifiied)

Process:

1) Open Chrome Browser and click 3 dots in top right corner > Help > About Google Chrome
--> this will show what version of chrome you have and if version is up to date

2) Go to the following website and download the driver for your current chrome version
    download: https://sites.google.com/a/chromium.org/chromedriver/downloads
    
3) Extract the files form the zip folder and Place in Downloads folder (or other known location)

Installation
############
To install the git version do

::
    git clone https://github.com/sdorkenw/MeshParty.git
    cd MeshParty
    pip install . -e --upgrade
    
    

Running
############
The pipeline application can be used in different ways and can be located in web_automation_tools/Applications/Eleox_Data_Fetch/Eleox_Data_Fetcher_vp1.py 

1) Used as command line python program 
::
    
    ...navigate to web_automation_tools/Applications/Eleox_Data_Fetch/
    python Eleox_Data_Fetcher_vp1.py [...command line arguments]
    
    
2) Accessed through the command in the command line
::

    pipeline_download [...command line arguments]


3) the data_fetch_pipeline() function and all helper functions copied into another repo/code and arguments sent through function


Arguments
############

::
    optional arguments:
    -h, --help            show this help message and exit
    -f EXPORT_FILEPATH, --export_filepath EXPORT_FILEPATH
                        csv filepath for the output of data
    -d DRIVER_EXE_PATH, --driver_exe_path DRIVER_EXE_PATH
                        path to where stored chrome driver exe
    -ret RETRIEVE_BUTTON_ID, --retrieve_button_id RETRIEVE_BUTTON_ID
                        the id of the retrieve button in the html source
    -down DOWNLOAD_BUTTON_ID, --download_button_id DOWNLOAD_BUTTON_ID
                        the id of the download button in the html source
    -v VISIBLE_BROWSER, --visible_browser VISIBLE_BROWSER
                        whether a browser window should pop up and perform the
                        scripted actions. Set to False for headerless
    -a APPEND_SOURCE, --append_source APPEND_SOURCE
                        whether the url and the pipeline name should be
                        appended to the entries to show where entry was
                        fetched from
    -s RETRIEVE_SLEEP_SECONDS, --retrieve_sleep_seconds RETRIEVE_SLEEP_SECONDS
                        how long the program will sleep after activating the
                        retrieve button (to help if takes long time to buffer)
    -b BASE_URL, --base_url BASE_URL
                        what webpage to start from
    -cat_n CATEGORY_CLASS_NAME, --category_class_name CATEGORY_CLASS_NAME
                        the class name from the html source to which signal
                        which tags to search for in finding categories
    -cat CATEGORIES [CATEGORIES ...], --categories CATEGORIES [CATEGORIES ...]
                        the pipelines to pull data from (listed in the
                        dropdown tabs of webpage). Currently only supports one
                        pipeline input specified with str
    
Examples: 

::
    # if the chrome driver exe is in your downloads folder in a folder called chromedriver_win32
    pipeline_download -f download.csv -d chromedriver_win32
    
    # if you wanted all of the pipelines in Midstream
    pipeline_download -cat Midstream
    