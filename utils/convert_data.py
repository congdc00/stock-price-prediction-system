# Convert data from download
import pandas as pd
import csv

if __name__ == "__main__":
    name_file_csv = "./data/test1.csv"
    df = pd.read_csv(name_file_csv)

    new_data = []
    print("Step 1 ...")
    for row in df.iterrows():
        content = row[1]

        time_tmp = content["Ngày"].split("/")
        datetime_now = time_tmp[2] + "-" + time_tmp[1] + "-" + time_tmp[0]

        close = content["Lần cuối"]
        time_open = content["Mở"]
        high = content["Cao"]
        low = content["Thấp"]
        try:
            volumn = content["KL"].replace("K", "")
            volumn = "{:.2f}".format(float(volumn)/1000)
            volumn = volumn+"M"
        except:
            volumn = content["KL"]
        percent_change = content["% Thay đổi"]
        
        new_content = [datetime_now, close, time_open, high, low, volumn, percent_change]
        new_data.append(new_content)
        
    print("Step 2 ...")
    new_csv = "./data/VN30_full.csv"
    with open(new_csv, 'w') as f:
        # create the csv writer
        writer = csv.writer(f)
        title = ["Date", "Close", "Open", "High", "Low", "Volumn", "Percent Change"]
        writer.writerow(title)
        for content in new_data:
            writer.writerow(content)
    # with open(name_file_csv, 'o') as f: